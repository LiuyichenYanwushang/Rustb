//! Uniform weak-field Floquet harmonics from the projected position operator.

use super::*;
use std::collections::{BTreeMap, BTreeSet};

fn paired_support<'a, const DIM: usize>(
    rows: impl IntoIterator<Item = &'a Vec<isize>>,
) -> Result<Vec<Vec<isize>>> {
    let mut support = BTreeSet::new();
    for row in rows {
        let opposite = row
            .iter()
            .map(|r| r.checked_neg())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                TbError::Other("Floquet position translation cannot be negated".into())
            })?;
        support.insert(row.clone());
        support.insert(opposite);
    }
    let origin = vec![0; DIM];
    support.remove(&origin);
    Ok(std::iter::once(origin).chain(support).collect())
}

// (blocks, translation rows), without exact-zero blocks in intermediate products.
type Blocks = (Vec<Array2<Complex<f64>>>, Array2<isize>);
type Harmonics = BTreeMap<isize, RealSpaceBlockMap>;

fn nonzero(z: Complex<f64>) -> bool {
    z.re != 0.0 || z.im != 0.0
}

fn checked_rescale(x: f64, scaled: f64) -> Result<f64> {
    if scaled.is_finite() && (scaled != 0.0 || x == 0.0) {
        Ok(scaled)
    } else {
        Err(TbError::Other(
            "Floquet position coupling exceeds the representable scaling range".into(),
        ))
    }
}

fn field_product(a: Complex<f64>, b: Complex<f64>) -> Result<Complex<f64>> {
    let value = (a * b) * 0.5;
    if !nonzero(value) && nonzero(a) && nonzero(b) {
        return Err(TbError::Other(
            "Floquet quadratic field coefficient underflow".into(),
        ));
    }
    Ok(value)
}

fn pack<const DIM: usize>(map: RealSpaceBlockMap) -> Blocks {
    let (rows, blocks): (Vec<_>, Vec<_>) = map
        .into_iter()
        .filter(|(_, block)| block.iter().copied().any(nonzero))
        .unzip();
    let mut support = Array2::zeros((rows.len(), DIM));
    for (mut dst, row) in support.outer_iter_mut().zip(rows) {
        dst.assign(&ArrayView1::from(&row));
    }
    (blocks, support)
}

fn add(target: &mut RealSpaceBlockMap, source: &Blocks, scale: Complex<f64>) {
    for (row, block) in source.1.outer_iter().zip(&source.0) {
        if block.iter().copied().any(nonzero) {
            target
                .entry(row.to_vec())
                .or_insert_with(|| Array2::zeros(block.raw_dim()))
                .scaled_add(scale, block);
        }
    }
}

fn commutator(a: &Blocks, b: &Blocks) -> Result<Blocks> {
    // A raw product lost in BLAS can later be amplified by W or 1/W. Refuse
    // extreme operand ranges rather than silently treating these terms as zero.
    let bounds = |blocks: &Blocks| {
        blocks
            .0
            .iter()
            .flatten()
            .flat_map(|z| [z.re.abs(), z.im.abs()])
            .filter(|&x| x != 0.0)
            .fold((f64::INFINITY, 0.0_f64), |(lo, hi), x| {
                (lo.min(x), hi.max(x))
            })
    };
    let (lo_a, hi_a) = bounds(a);
    let (lo_b, hi_b) = bounds(b);
    if hi_a != 0.0 && hi_b != 0.0 && (lo_a * lo_b < f64::MIN_POSITIVE || !(hi_a * hi_b).is_finite())
    {
        return Err(TbError::Other(
            "Floquet position commutator exceeds the representable product range".into(),
        ));
    }
    real_space_commutator_with_supports(&a.0, &a.1, &b.0, &b.1)
}

fn derivative(
    target: &mut RealSpaceBlockMap,
    source: &Blocks,
    direction: ArrayView1<'_, f64>,
    sign: f64,
) -> Result<()> {
    for (row, block) in source.1.outer_iter().zip(&source.0) {
        let distance: f64 = row.iter().zip(direction).map(|(&r, &d)| r as f64 * d).sum();
        if distance == 0.0 {
            continue;
        }
        let weight = sign * distance;
        let output = target
            .entry(row.to_vec())
            .or_insert_with(|| Array2::zeros(block.raw_dim()));
        for (dst, z) in output.iter_mut().zip(block) {
            // i times the Cartesian displacement, with a check before a
            // later frequency factor can amplify an underflowed derivative.
            *dst += Complex::new(
                checked_rescale(z.im, -weight * z.im)?,
                checked_rescale(z.re, weight * z.re)?,
            );
        }
    }
    Ok(())
}

// In the lattice gauge X_i = i∂_i + A_i(k), with A_i stored in rmatrix.
// D_i B = ∂_i B + i[B,A_i]; F_ij = ∂_i A_j − ∂_j A_i − i[A_i,A_j].
fn covariant_derivative<const DIM: usize>(
    source: &Blocks,
    position: &Blocks,
    direction: ArrayView1<'_, f64>,
) -> Result<Blocks> {
    let mut result = RealSpaceBlockMap::new();
    derivative(&mut result, source, direction, 1.0)?;
    add(&mut result, &commutator(source, position)?, Complex::i());
    Ok(pack::<DIM>(result))
}

fn field_coefficients<const DIM: usize>(
    drive: &FloquetDrive,
) -> Result<BTreeMap<i128, [Complex<f64>; DIM]>> {
    // The signed actual Fourier coefficients b_n satisfy b_-n = b_n*.
    // Widen first: both negating isize::MIN and summing two labels must work.
    let mut field = BTreeMap::<i128, [Complex<f64>; DIM]>::new();
    for mode in &drive.modes {
        let n = mode.harmonic as i128;
        if n == 0 {
            for (b, a) in field
                .entry(0)
                .or_insert([Complex::ZERO; DIM])
                .iter_mut()
                .zip(&mode.a_complex)
            {
                b.re += a.re;
            }
        } else {
            for (b, a) in field
                .entry(n.abs())
                .or_insert([Complex::ZERO; DIM])
                .iter_mut()
                .zip(&mode.a_complex)
            {
                let im = if n < 0 { -a.im } else { a.im };
                *b += Complex::new(
                    checked_rescale(a.re, a.re * 0.5)?,
                    checked_rescale(im, im * 0.5)?,
                );
            }
        }
    }
    field.retain(|_, a| a.iter().copied().any(nonzero));
    if field.values().flatten().any(|a| !a.is_finite()) {
        return Err(TbError::Other(
            "Floquet coherent position-drive amplitude overflow".into(),
        ));
    }
    let negative: Vec<_> = field
        .iter()
        .filter(|&(&n, _)| n > 0)
        .map(|(&n, a)| (-n, a.map(|z| z.conj())))
        .collect();
    field.extend(negative);
    Ok(field)
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    fn bare_floquet_blocks(&self) -> RealSpaceBlockMap {
        self.hamR
            .outer_iter()
            .zip(self.ham.outer_iter())
            .map(|(r, t)| (r.to_vec(), t.to_owned()))
            .collect()
    }

    pub(super) fn validate_floquet_position(&self) -> Result<()> {
        let indices: BTreeMap<_, _> = self
            .hamR
            .outer_iter()
            .enumerate()
            .map(|(i, r)| (r.to_vec(), i))
            .collect();
        if indices.len() != self.hamR.nrows() {
            return Err(TbError::Other(
                "Floquet position input has duplicate translations".into(),
            ));
        }
        let r = self.rmatrix.as_array4();
        for (row, &ir) in &indices {
            let opposite = row
                .iter()
                .map(|x| x.checked_neg())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    TbError::Other("Floquet position translation cannot be negated".into())
                })?;
            let partner = indices.get(&opposite);
            let mismatch = |z: Complex<f64>, y: Complex<f64>| {
                let scale = z.re.abs().max(z.im.abs()).max(y.re.abs()).max(y.im.abs());
                (z.re - y.re).abs().max((z.im - y.im).abs()) > 1e-12 * scale
            };
            for ((i, j), &z) in self.ham.index_axis(Axis(0), ir).indexed_iter() {
                let y = partner.map_or(Complex::ZERO, |&jr| self.ham[[jr, j, i]].conj());
                if mismatch(z, y) {
                    return Err(TbError::InvalidModelInvariant {
                        invariant: "hermitian_hamiltonian",
                        message: format!(
                            "non-Hermitian hopping at translation {row:?}, state pair ({i}, {j})"
                        ),
                    });
                }
            }
            for ((d, i, j), &z) in r.index_axis(Axis(0), ir).indexed_iter() {
                let y = partner.map_or(Complex::ZERO, |&jr| r[[jr, d, j, i]].conj());
                if mismatch(z, y) {
                    return Err(TbError::InvalidModelInvariant {
                        invariant: "hermitian_position_matrix",
                        message: format!(
                            "non-Hermitian position block at translation {row:?}, component {d}, state pair ({i}, {j})"
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    fn position_harmonics(
        &self,
        drive: &FloquetDrive,
        linear_max: isize,
        quadratic_max: isize,
    ) -> Result<(Harmonics, Harmonics)> {
        let mut field = field_coefficients::<DIM>(drive)?;
        let mut linear = Harmonics::new();
        let mut quadratic = Harmonics::new();
        if field.is_empty() {
            return Ok((linear, quadratic));
        }

        // Balance units before quadratic products: form (s_i D_i) and
        // (s_i X_i) using the largest field component s_i on each axis.
        // Squaring either the unscaled position or the field separately can
        // underflow/overflow even for a small, representable product a·X.
        let scale: [f64; DIM] = std::array::from_fn(|i| {
            field
                .values()
                .map(|a| a[i].re.abs().max(a[i].im.abs()))
                .fold(0.0, f64::max)
        });
        for a in field.values_mut() {
            for (z, &s) in a.iter_mut().zip(&scale) {
                if s != 0.0 {
                    *z = Complex::new(
                        checked_rescale(z.re, z.re / s)?,
                        checked_rescale(z.im, z.im / s)?,
                    );
                }
            }
        }
        let mut lattice = self.lat.clone();
        for (mut column, &s) in lattice.columns_mut().into_iter().zip(&scale) {
            for x in &mut column {
                *x = if s == 0.0 {
                    0.0
                } else {
                    checked_rescale(*x, *x * s)?
                };
            }
        }

        // We transform the projected length-gauge Hamiltonian with U=e^{i a·X}.
        // The quadratic term is ½ a_i a_j D_i D_j H − ħ/2 a_i dot(a)_j F_ij.
        // Compute the scalar Fourier weights first, then build each real-space
        // derivative/curvature only once, independently of the mode count.
        let mut weights =
            BTreeMap::<isize, ([[Complex<f64>; DIM]; DIM], [[Complex<f64>; DIM]; DIM])>::new();
        for (&p, left) in &field {
            for (&q, right) in
                field.range((-(quadratic_max as i128) - p)..=((quadratic_max as i128) - p))
            {
                let (hessian, curvature) = weights
                    .entry((p + q) as isize)
                    .or_insert(([[Complex::ZERO; DIM]; DIM], [[Complex::ZERO; DIM]; DIM]));
                for i in 0..DIM {
                    for j in 0..DIM {
                        hessian[i][j] += field_product(left[i], right[j])?;
                        if i < j && p < q {
                            // Pair temporal orders (p,q) and (q,p), using F_ji = −F_ij.
                            let wedge = field_product(left[i], right[j])?
                                - field_product(right[i], left[j])?;
                            if nonzero(wedge) {
                                let weighted = Complex::<f64>::i() * wedge * ((q - p) as f64);
                                curvature[i][j] += Complex::new(
                                    checked_rescale(weighted.re, weighted.re * drive.omega0_ev)?,
                                    checked_rescale(weighted.im, weighted.im * drive.omega0_ev)?,
                                );
                            }
                        }
                    }
                }
            }
        }

        let h = pack::<DIM>(self.bare_floquet_blocks());
        let r = self.rmatrix.as_array4();
        let position: Vec<_> = (0..DIM)
            .map(|i| -> Result<_> {
                let s = scale[i];
                let mut blocks = RealSpaceBlockMap::new();
                if s != 0.0 {
                    for (row, input) in self
                        .hamR
                        .outer_iter()
                        .zip(r.index_axis(Axis(1), i).outer_iter())
                    {
                        if input.iter().copied().any(nonzero) {
                            let mut block = input.to_owned();
                            for z in &mut block {
                                *z = Complex::new(
                                    checked_rescale(z.re, z.re * s)?,
                                    checked_rescale(z.im, z.im * s)?,
                                );
                            }
                            blocks.insert(row.to_vec(), block);
                        }
                    }
                }
                Ok(pack::<DIM>(blocks))
            })
            .collect::<Result<_>>()?;
        for j in 0..DIM {
            let needs_linear = field
                .range((-(linear_max as i128))..=(linear_max as i128))
                .any(|(_, a)| nonzero(a[j]));
            let needs_hessian = weights
                .values()
                .any(|(hessian, _)| (0..DIM).any(|i| nonzero(hessian[i][j])));
            if !needs_linear && !needs_hessian {
                continue;
            }
            let gradient = covariant_derivative::<DIM>(&h, &position[j], lattice.column(j))?;
            for (&n, a) in field.range((-(linear_max as i128))..=(linear_max as i128)) {
                if nonzero(a[j]) {
                    add(linear.entry(n as isize).or_default(), &gradient, -a[j]);
                }
            }
            for i in 0..DIM {
                if weights.values().any(|(hessian, _)| nonzero(hessian[i][j])) {
                    let second =
                        covariant_derivative::<DIM>(&gradient, &position[i], lattice.column(i))?;
                    for (&n, (hessian, _)) in &weights {
                        if nonzero(hessian[i][j]) {
                            add(quadratic.entry(n).or_default(), &second, hessian[i][j]);
                        }
                    }
                }
            }
        }
        for i in 0..DIM {
            for j in (i + 1)..DIM {
                if !weights
                    .values()
                    .any(|(_, curvature)| nonzero(curvature[i][j]))
                {
                    continue;
                }
                let mut f = RealSpaceBlockMap::new();
                derivative(&mut f, &position[j], lattice.column(i), 1.0)?;
                derivative(&mut f, &position[i], lattice.column(j), -1.0)?;
                add(
                    &mut f,
                    &commutator(&position[i], &position[j])?,
                    -Complex::i(),
                );
                let f = pack::<DIM>(f);
                for (&n, (_, curvature)) in &weights {
                    if nonzero(curvature[i][j]) {
                        add(quadratic.entry(n).or_default(), &f, curvature[i][j]);
                    }
                }
            }
        }
        Ok((linear, quadratic))
    }

    // Sparse Fourier blocks for Sambe; the union of their supports is built
    // before the enlarged model is allocated.
    pub(super) fn position_sambe_harmonics(
        &self,
        drive: &FloquetDrive,
        harmonic_max: isize,
    ) -> Result<Harmonics> {
        let (mut linear, quadratic) = self.position_harmonics(drive, harmonic_max, harmonic_max)?;
        for (n, blocks) in quadratic {
            let target = linear.entry(n).or_default();
            *target = merge_real_space_block_maps(std::mem::take(target), blocks);
        }
        // Accumulate the complete field correction before adding the bare
        // Hamiltonian, as in the time-averaged effective-model path.
        let zero = linear.entry(0).or_default();
        *zero = merge_real_space_block_maps(std::mem::take(zero), self.bare_floquet_blocks());
        Ok(linear)
    }

    pub(super) fn position_sambe_cache(
        &self,
        drive: &FloquetDrive,
        harmonic_max: isize,
        total: usize,
    ) -> Result<(Array2<isize>, FloquetHarmonicCache)> {
        let harmonics = self.position_sambe_harmonics(drive, harmonic_max)?;
        let support = paired_support::<DIM>(harmonics.values().flat_map(|blocks| blocks.keys()))?;
        let harmonic_count = (2 * harmonic_max + 1) as usize;
        let nsta = self.nsta();
        validate_sambe_layout::<DIM, R>(support.len(), total, nsta, harmonic_count)?;
        let mut ham_r = Array2::zeros((support.len(), DIM));
        let mut blocks = Array4::zeros((harmonic_count, support.len(), nsta, nsta));
        let mut by_row = BTreeMap::new();
        for (ir, row) in support.into_iter().enumerate() {
            ham_r.row_mut(ir).assign(&ArrayView1::from(&row));
            by_row.insert(row, ir);
        }
        for (n, harmonic) in harmonics {
            for (row, block) in harmonic {
                blocks
                    .slice_mut(s![(n + harmonic_max) as usize, by_row[&row], .., ..])
                    .assign(&block);
            }
        }
        Ok((
            ham_r,
            FloquetHarmonicCache {
                harmonic_min: -harmonic_max,
                harmonic_max,
                blocks,
            },
        ))
    }

    // This is the bare position operator in the returned basis, not an
    // observable transformed by the optical gauge and Floquet micromotion.
    pub(super) fn floquet_output_position(
        &self,
        support: &Array2<isize>,
        total: usize,
        photon_basis: Option<&[Vec<usize>]>,
    ) -> Result<R> {
        if !R::HAS_RMATRIX {
            return Ok(self.rmatrix.clone());
        }
        let nr = support.nrows();
        [nr, DIM, total, total]
            .into_iter()
            .try_fold(size_of::<Complex<f64>>(), usize::checked_mul)
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| {
                TbError::Other("Floquet position array size exceeds the addressable range".into())
            })?;
        let mut rmatrix = Array4::zeros((nr, DIM, total, total));
        let input: BTreeMap<_, _> = self
            .hamR
            .outer_iter()
            .enumerate()
            .map(|(ir, row)| (row.to_vec(), ir))
            .collect();
        for (ir, row) in support.outer_iter().enumerate() {
            let Some(&old_ir) = input.get(&row.to_vec()) else {
                continue;
            };
            let source = self.rmatrix.as_array4().index_axis(Axis(0), old_ir);
            if let Some(sectors) = photon_basis {
                for sector in sectors {
                    for ((d, i, j), &value) in source.indexed_iter() {
                        rmatrix[[ir, d, sector[i], sector[j]]] = value;
                    }
                }
            } else {
                rmatrix.index_axis_mut(Axis(0), ir).assign(&source);
            }
        }
        Ok(R::from_array(rmatrix))
    }

    pub(super) fn position_effective_corrections(
        &self,
        drive: &FloquetDrive,
        order: usize,
        harmonic_max: isize,
    ) -> Result<RealSpaceBlockMap> {
        let (mut linear, mut quadratic) =
            self.position_harmonics(drive, if order == 0 { 0 } else { harmonic_max }, 0)?;
        let mut result = RealSpaceBlockMap::new();
        for (row, block) in linear
            .remove(&0)
            .into_iter()
            .chain(quadratic.remove(&0))
            .flatten()
        {
            result
                .entry(row)
                .and_modify(|dst| *dst += &block)
                .or_insert(block);
        }
        // Strict field order: only the linear harmonics enter van Vleck
        // products, and the middle block at frequency order two is bare H.
        let h = (order == 2).then(|| pack::<DIM>(self.bare_floquet_blocks()));
        let mut linear: BTreeMap<_, _> = linear
            .into_iter()
            .map(|(n, blocks)| (n, pack::<DIM>(blocks)))
            .collect();
        while let Some((n, plus)) = linear.pop_last() {
            if n < 0 {
                break;
            }
            let minus = linear
                .remove(&-n)
                .expect("signed position harmonics are paired");
            if plus.0.is_empty() || minus.0.is_empty() {
                continue;
            }
            let inverse = (1.0 / n as f64) / drive.omega0_ev;
            let (first, support) = commutator(&minus, &plus)?;
            accumulate_scaled_real_space_blocks(&mut result, &first, &support, inverse)?;
            if let Some(h) = &h {
                for (a, b) in [(&minus, &plus), (&plus, &minus)] {
                    let inner = commutator(h, b)?;
                    let (mut outer, support) = commutator(a, &inner)?;
                    // Scale the matrix before applying the second inverse. The
                    // squared scalar alone can underflow even for a finite term.
                    for z in outer.iter_mut().flatten() {
                        if nonzero(*z) {
                            *z = if inverse > 1.0 {
                                (*z * (0.5 * inverse)) * inverse
                            } else {
                                ((*z * inverse) * inverse) * 0.5
                            };
                        }
                    }
                    accumulate_scaled_real_space_blocks(&mut result, &outer, &support, 1.0)?;
                }
            }
        }
        Ok(result)
    }

    pub(super) fn position_effective_output(
        &self,
        corrections: RealSpaceBlockMap,
    ) -> Result<Model<SPIN, DIM, R>> {
        let blocks = merge_real_space_block_maps(self.bare_floquet_blocks(), corrections);
        let support = paired_support::<DIM>(blocks.keys())?;
        let nsta = self.nsta();
        let mut ham_r = Array2::zeros((support.len(), DIM));
        let mut ham = Array3::zeros((support.len(), nsta, nsta));
        for (ir, row) in support.into_iter().enumerate() {
            ham_r.row_mut(ir).assign(&ArrayView1::from(&row));
            if let Some(block) = blocks.get(&row) {
                ham.index_axis_mut(Axis(0), ir).assign(block);
            }
        }
        place_origin_first(&mut ham, &mut ham_r)?;
        enforce_real_space_hermiticity(&mut ham, &ham_r)?;
        self.floquet_effective_output(ham, ham_r)
    }
}
