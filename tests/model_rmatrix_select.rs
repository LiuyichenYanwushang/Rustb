fn fixture() -> ndarray::Array4<num_complex::Complex<f64>> {
    use ndarray::Array4;
    use num_complex::Complex;

    Array4::from_shape_fn((2, 1, 2, 2), |(r, _, i, j)| {
        let value = (1 + 4 * r + 2 * i + j) as f64;
        Complex::new(value, -value)
    })
}

#[test]
fn generic_select_preserves_rmatrix_type() {
    use Rustb::{HasRMatrix, NoRMatrix, RMatrixData};
    use ndarray::Axis;
    use num_complex::Complex;

    fn select<R: RMatrixData>(matrix: &R, axis: Axis, indices: &[usize]) -> R {
        matrix.select(axis, indices)
    }

    let original = fixture();
    let source = HasRMatrix(original.clone());
    let selected: HasRMatrix = select(&source, Axis(0), &[1, 0, 1]);
    assert_eq!(selected.dim(), (3, 1, 2, 2));
    assert_eq!(
        selected.iter().copied().collect::<Vec<_>>(),
        [5., 6., 7., 8., 1., 2., 3., 4., 5., 6., 7., 8.].map(|n| Complex::new(n, -n))
    );
    let empty: HasRMatrix = select(&source, Axis(0), &[]);
    assert_eq!(empty.dim(), (0, 1, 2, 2));
    assert!(empty.is_empty());
    assert_eq!(source.0, original);

    let absent = NoRMatrix;
    let _: NoRMatrix = select(&absent, Axis(0), &[1, 0, 1]);
    let _: NoRMatrix = select(&absent, Axis(0), &[]);
}

#[test]
fn concrete_select_with_trait_returns_has_rmatrix() {
    use Rustb::{HasRMatrix, RMatrixData};
    use ndarray::Axis;
    use num_complex::Complex;

    let original = fixture();
    let source = HasRMatrix(original.clone());
    let selected: HasRMatrix = source.select(Axis(2), &[1, 0, 1]);
    assert_eq!(selected.dim(), (2, 1, 3, 2));
    assert_eq!(
        selected.iter().copied().collect::<Vec<_>>(),
        [3., 4., 1., 2., 3., 4., 7., 8., 5., 6., 7., 8.].map(|n| Complex::new(n, -n))
    );
    let empty: HasRMatrix = source.select(Axis(2), &[]);
    assert_eq!(empty.dim(), (2, 1, 0, 2));
    assert!(empty.is_empty());
    assert_eq!(source.0, original);
}

#[test]
fn concrete_select_without_trait_returns_ndarray() {
    // Keep RMatrixData out of this scope so method lookup uses ndarray via Deref.
    use Rustb::HasRMatrix;
    use ndarray::{Array4, Axis};
    use num_complex::Complex;

    let original = fixture();
    let source = HasRMatrix(original.clone());
    let selected: Array4<Complex<f64>> = source.select(Axis(3), &[1, 0, 1]);
    assert_eq!(selected.dim(), (2, 1, 2, 3));
    assert_eq!(
        selected.iter().copied().collect::<Vec<_>>(),
        [2., 1., 2., 4., 3., 4., 6., 5., 6., 8., 7., 8.].map(|n| Complex::new(n, -n))
    );
    let empty: Array4<Complex<f64>> = source.select(Axis(3), &[]);
    assert_eq!(empty.dim(), (2, 1, 2, 0));
    assert!(empty.is_empty());
    assert_eq!(source.0, original);
}
