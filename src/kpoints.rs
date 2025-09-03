use ndarray::{Array1,Array2,Array3,Axis};
use crate::generics::usefloat;
use crate::error::{TbError, Result};

#[allow(non_snake_case)]
#[inline(always)]
pub fn gen_kmesh<T>(k_mesh: &Array1<usize>) -> Result<Array2<T>>
where
    T: usefloat + std::ops::Div<Output = T>,
{
    let dim: usize = k_mesh.len();
    let mut nk: usize = 1;
    for i in 0..dim {
        nk *= k_mesh[[i]];
    }
    fn gen_kmesh_arr<T>(k_mesh: &Array1<usize>, r0: usize, mut usek: Array1<T>) -> Array2<T>
    where
        T: usefloat + std::ops::Div<Output = T>,
    {
        let dim: usize = k_mesh.len();
        let mut kvec = Array2::<T>::zeros((0, dim));
        if r0 == 0 {
            for i in 0..(k_mesh[[r0]]) {
                let mut usek = Array1::<T>::zeros(dim);
                usek[[r0]] = T::from(i) / T::from(k_mesh[[r0]]);
                let k0: Array2<T> = gen_kmesh_arr(&k_mesh, r0 + 1, usek);
                kvec.append(Axis(0), k0.view()).unwrap();
            }
            return kvec;
        } else if r0 < k_mesh.len() - 1 {
            for i in 0..(k_mesh[[r0]]) {
                let mut kk = usek.clone();
                kk[[r0]] = T::from(i) / T::from(k_mesh[[r0]]);
                let k0: Array2<T> = gen_kmesh_arr(&k_mesh, r0 + 1, kk);
                kvec.append(Axis(0), k0.view()).unwrap();
            }
            return kvec;
        } else {
            for i in 0..(k_mesh[[r0]]) {
                usek[[r0]] = T::from(i) / T::from(k_mesh[[r0]]);
                kvec.push_row(usek.view()).unwrap();
            }
            return kvec;
        }
    }
    let mut usek = Array1::<T>::zeros(dim);
    Ok(gen_kmesh_arr(&k_mesh, 0, usek))
}
#[allow(non_snake_case)]
#[inline(always)]
pub fn gen_krange<T>(k_mesh: &Array1<usize>) -> Result<Array3<T>>
where
    T: usefloat + std::ops::Div<Output = T>,
{
    let dim_r = k_mesh.len();
    let mut k_range = Array3::<T>::zeros((0, dim_r, 2));
    match dim_r {
        1 => {
            for i in 0..k_mesh[[0]] {
                let mut k = Array2::<T>::zeros((dim_r, 2));
                k[[0, 0]] = T::from(i) / T::from(k_mesh[[0]]);
                k[[0, 1]] = T::from(i + 1) / T::from(k_mesh[[0]]);
                k_range.push(Axis(0), k.view()).unwrap();
            }
        }
        2 => {
            for i in 0..k_mesh[[0]] {
                for j in 0..k_mesh[[1]] {
                    let mut k = Array2::<T>::zeros((dim_r, 2));
                    k[[0, 0]] = T::from(i) / T::from(k_mesh[[0]]);
                    k[[0, 1]] = T::from(i + 1) / T::from(k_mesh[[0]]);
                    k[[1, 0]] = T::from(j) / T::from(k_mesh[[1]]);
                    k[[1, 1]] = T::from(j + 1) / T::from(k_mesh[[1]]);
                    k_range.push(Axis(0), k.view()).unwrap();
                }
            }
        }
        3 => {
            for i in 0..k_mesh[[0]] {
                for j in 0..k_mesh[[1]] {
                    for ks in 0..k_mesh[[2]] {
                        let mut k = Array2::<T>::zeros((dim_r, 2));
                        k[[0, 0]] = T::from(i) / T::from(k_mesh[[0]]);
                        k[[0, 1]] = T::from(i + 1) / T::from(k_mesh[[0]]);
                        k[[1, 0]] = T::from(j) / T::from(k_mesh[[1]]);
                        k[[1, 1]] = T::from(j + 1) / T::from(k_mesh[[1]]);
                        k[[2, 0]] = T::from(ks) / T::from(k_mesh[[2]]);
                        k[[2, 1]] = T::from(ks + 1) / T::from(k_mesh[[2]]);
                        k_range.push(Axis(0), k.view()).unwrap();
                    }
                }
            }
        }
        _ => {
            return Err(TbError::InvalidDimension {
                dim: dim_r,
                supported: vec![1, 2, 3],
            });
        }
    };
    Ok(k_range)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array,Array2};

    #[test]
    fn test_gen_kmesh() {
        // Test basic 2x2 kmesh generation
        let kmesh:Array2<f64> = gen_kmesh(&array![2, 2]).unwrap();
        assert_eq!(kmesh.shape(), &[4, 2]); // 4 points in 2D
        // Add more assertions based on expected output
    }

}
