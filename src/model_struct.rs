use ndarray::*;
use num_complex::Complex;
use crate::Atom;
use crate::Model;




impl Model{
    #[inline(always)]
    pub fn atom_position(&self)->Array2<f64>{
        let mut atom_position=Array2::zeros((self.natom(),self.dim_r));
        atom_position.outer_iter_mut().zip(self.atoms.iter()).for_each(|(mut atom_p,atom)|
            {
                atom_p.assign(&atom.position());
            }
        );
        atom_position        
    }
    #[inline(always)]
    pub fn atom_list(&self)->Vec<usize>{
        let mut atom_list=Vec::new();
        for a in self.atoms.iter(){
            atom_list.push(a.norb());
        }
        atom_list
    }
    #[inline(always)]
    pub fn natom(&self)->usize{
        self.atoms.len()
    }
    #[inline(always)]
    pub fn norb(&self)->usize{
        self.orb.nrows()
    }
    #[inline(always)]
    pub fn nsta(&self)->usize{
        if self.spin{
            2*self.norb()
        }else{
            self.norb()
        }
    }
    /*
    pub fn orb_angular(&self)->Array3::<Complex<f64>>{
        ///这个函数输出 $\bra{m,\bm k}L\ket{n,\bm k}$ 矩阵, 这里 $\ket{n,\bm k}$
        ///是根据轨道的projection 得到这个基函下的表示
        ///这个表示是依据每个原子来构造的, 所以是一个块对角的矩阵
        let mut i=0;
        let mut L=Array3::zeros((self.dim_r,self.norb(),self.norb()));
        let mut Lx=Array2::zeros((self.norb(),self.norb()));
        let mut Ly=Array2::zeros((self.norb(),self.norb()));
        let mut Lz=Array2::zeros((self.norb(),self.norb()));
        for r in 0..self.dim_r{
            for a in self.atom.iter(){
                let norb=atom.norb();
                let mut proj=Array2::zeros((norb,norb));
                for m in 0..norb{
                    for n in 0..norb{
                    }
                }
                i+=norb;
            }
        }
    }
    */
}
