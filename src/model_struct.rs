use crate::Atom;
use crate::Model;
use crate::atom_struct::AtomType;
use crate::error::{TbError, Result};
use ndarray::*;
use num_complex::Complex;

impl Model {
    #[inline(always)]
    pub fn atom_position(&self) -> Array2<f64> {
        let mut atom_position = Array2::zeros((self.natom(), self.dim_r));
        atom_position
            .outer_iter_mut()
            .zip(self.atoms.iter())
            .for_each(|(mut atom_p, atom)| {
                atom_p.assign(&atom.position());
            });
        atom_position
    }
    pub fn dim_r(&self) -> usize {
        self.lat.nrows()
    }
    #[inline(always)]
    pub fn atom_list(&self) -> Vec<usize> {
        let mut atom_list = Vec::new();
        for a in self.atoms.iter() {
            atom_list.push(a.norb());
        }
        atom_list
    }
    #[inline(always)]
    pub fn atom_type(&self) -> Vec<AtomType> {
        let mut atom_type = Vec::new();
        for a in self.atoms.iter() {
            atom_type.push(a.atom_type());
        }
        atom_type
    }
    #[inline(always)]
    pub fn nR(&self) -> usize {
        self.hamR.nrows()
    }

    #[inline(always)]
    pub fn natom(&self) -> usize {
        self.atoms.len()
    }
    #[inline(always)]
    pub fn norb(&self) -> usize {
        self.orb.nrows()
    }
    #[inline(always)]
    pub fn nsta(&self) -> usize {
        if self.spin {
            2 * self.norb()
        } else {
            self.norb()
        }
    }
    #[inline(always)]
    pub fn orb_angular(&self) -> Result<Array3<Complex<f64>>> {
        //!这个函数输出 $\bra{m,\bm k}L\ket{n,\bm k}$ 矩阵, 这里 $\ket{n,\bm k}$
        //!是根据轨道的projection 得到这个基函下的表示
        //!这个表示是依据每个原子来构造的, 所以是一个块对角的矩阵
        //!
        //!目前根据最新的轨道流公式, 这个已经废弃不使用了, 求轨道角动量见
        //!
        //!orbital_angular_momentom 函数
        let li = Complex::i() * 1.0;
        let mut i = 0;
        let mut L = Array3::<Complex<f64>>::zeros((self.dim_r(), self.norb(), self.norb()));
        let mut Lx = Array2::<Complex<f64>>::zeros((self.norb(), self.norb()));
        let mut Ly = Array2::<Complex<f64>>::zeros((self.norb(), self.norb()));
        let mut Lz = Array2::<Complex<f64>>::zeros((self.norb(), self.norb()));
        //开始构造在量子数为基的情况下的 Lz, L+ 和 L-.
        let mut Lz_orig = Array2::<Complex<f64>>::zeros((16, 16));
        Lz_orig
            .slice_mut(s![1..4, 1..4])
            .assign(&Array2::from_diag(&array![-1.0, 0.0, 1.0]).mapv(|x| Complex::new(x, 0.0)));
        Lz_orig.slice_mut(s![4..9, 4..9]).assign(
            &Array2::from_diag(&array![-2.0, -1.0, 0.0, 1.0, 2.0]).mapv(|x| Complex::new(x, 0.0)),
        );
        Lz_orig.slice_mut(s![9..16, 9..16]).assign(
            &Array2::from_diag(&array![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
                .mapv(|x| Complex::new(x, 0.0)),
        );
        let mut Lup_orig = Array2::<Complex<f64>>::zeros((16, 16));
        let mut Ldn_orig = Array2::<Complex<f64>>::zeros((16, 16));
        for l in 0..4 {
            for (m0, m) in (-(l as isize)..=l as isize).enumerate() {
                let i = l * l + m0;
                if m + 1 > l as isize && m - 1 < -(l as isize) {
                    continue;
                } else if m + 1 > l as isize {
                    let l = l as f64;
                    let m = m as f64;
                    Ldn_orig[[i - 1, i]] =
                        Complex::new((l * (l + 1.0) - m * (m - 1.0)).sqrt(), 0.0);
                } else if m - 1 < -(l as isize) {
                    let l = l as f64;
                    let m = m as f64;
                    Lup_orig[[i + 1, i]] =
                        Complex::new((l * (l + 1.0) - m * (m + 1.0)).sqrt(), 0.0);
                } else {
                    let l = l as f64;
                    let m = m as f64;
                    Ldn_orig[[i - 1, i]] =
                        Complex::new((l * (l + 1.0) - m * (m - 1.0)).sqrt(), 0.0);
                    Lup_orig[[i + 1, i]] =
                        Complex::new((l * (l + 1.0) - m * (m + 1.0)).sqrt(), 0.0);
                }
            }
        }
        //Lx=L+ + L-, Ly=-i( L+ - L-)
        let Lx_orig = &Lup_orig + &Ldn_orig;
        let Ly_orig = -li * (&Lup_orig - &Ldn_orig);
        //接下来我们要根据我们的轨道基函数写出我们的轨道角动量的矩阵
        //轨道角动量矩阵是按照原子进行分块对角化的
        //所以, 我们作循环的时候也要先按照原子进行分块对角化
        let mut a = 0;
        for atom0 in self.atoms.iter() {
            for i in a..a + atom0.norb() {
                let proj_i: Array1<Complex<f64>> = self.orb_projection[i]
                    .to_quantum_number()?
                    .mapv(|x: Complex<f64>| x.conj());
                for j in a..a + atom0.norb() {
                    let proj_j: Array1<Complex<f64>> = self.orb_projection[j].to_quantum_number()?;
                    L[[0, i, j]] = proj_i.dot(&Lx_orig.dot(&proj_j));
                    L[[1, i, j]] = proj_i.dot(&Ly_orig.dot(&proj_j));
                    L[[2, i, j]] = proj_i.dot(&Lz_orig.dot(&proj_j));
                }
            }
            a += atom0.norb();
        }
        Ok(L)
    }
}
