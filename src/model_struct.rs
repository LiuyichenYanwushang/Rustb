use ndarray::*;
use num_complex::Complex;
use crate::atom;
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
}
