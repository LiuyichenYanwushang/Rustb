use crate::atom_struct::{OrbProj,Atom,AtomType};
use spglib::cell::Cell;
use spglib::dataset::Dataset;
use crate::Model;


impl From<AtomType> for i32 {
    fn from(atom: AtomType) -> Self {
        match atom {
            AtomType::H  =>  1,
            AtomType::He =>  2,
            AtomType::Li =>  3,
            AtomType::Be =>  4,
            AtomType::B  =>  5,
            AtomType::C  =>  6,
            AtomType::N  =>  7,
            AtomType::O  =>  8,
            AtomType::F  =>  9,
            AtomType::Ne => 10,
            AtomType::Na => 11,
            AtomType::Mg => 12,
            AtomType::Al => 13,
            AtomType::Si => 14,
            AtomType::P  => 15,
            AtomType::S  => 16,
            AtomType::Cl => 17,
            AtomType::Ar => 18,
            AtomType::K  => 19,
            AtomType::Ca => 20,
            AtomType::Sc => 21,
            AtomType::Ti => 22,
            AtomType::V  => 23,
            AtomType::Cr => 24,
            AtomType::Mn => 25,
            AtomType::Fe => 26,
            AtomType::Co => 27,
            AtomType::Ni => 28,
            AtomType::Cu => 29,
            AtomType::Zn => 30,
            AtomType::Ga => 31,
            AtomType::Ge => 32,
            AtomType::As => 33,
            AtomType::Se => 34,
            AtomType::Br => 35,
            AtomType::Kr => 36,
            AtomType::Rb => 37,
            AtomType::Sr => 38,
            AtomType::Y  => 39,
            AtomType::Zr => 40,
            AtomType::Nb => 41,
            AtomType::Mo => 42,
            AtomType::Tc => 43,
            AtomType::Ru => 44,
            AtomType::Rh => 45,
            AtomType::Pd => 46,
            AtomType::Ag => 47,
            AtomType::Cd => 48,
            AtomType::In => 49,
            AtomType::Sn => 50,
            AtomType::Sb => 51,
            AtomType::Te => 52,
            AtomType:: I => 53,
            AtomType::Xe => 54,
            AtomType::Cs => 55,
            AtomType::Ba => 56,
            AtomType::La => 57,
            AtomType::Ce => 58,
            AtomType::Pr => 59,
            AtomType::Nd => 60,
            AtomType::Pm => 61,
            AtomType::Sm => 62,
            AtomType::Eu => 63,
            AtomType::Gd => 64,
            AtomType::Tb => 65,
            AtomType::Dy => 66,
            AtomType::Ho => 67,
            AtomType::Er => 68,
            AtomType::Tm => 69,
            AtomType::Yb => 70,
            AtomType::Lu => 71,
            AtomType::Hf => 72,
            AtomType::Ta => 73,
            AtomType:: W => 74,
            AtomType::Re => 75,
            AtomType::Os => 76,
            AtomType::Ir => 77,
            AtomType::Pt => 78,
            AtomType::Au => 79,
            AtomType::Hg => 80,
            AtomType::Tl => 81,
            AtomType::Pb => 82,
            AtomType::Bi => 83,
            AtomType::Po => 84,
            AtomType::At => 85,
            AtomType::Rn => 86,
            AtomType::Fr => 87,
            AtomType::Ra => 88,
        }
    }
}

impl Model{
    pub fn space_group(&self)->Dataset{
        let mut lattice=[[0.0;3];3];
        for i in 0..3{
            lattice[i][i]=1.0;
        }
        let mut position=Vec::new();
        let mut atomtype=Vec::new();
        for i in 0..self.dim_r(){
            for j in 0..self.dim_r(){
                lattice[i][j]=self.lat[[i,j]];
            }
        }
        for atom in self.atoms.iter(){
            let mut pos=[0.0;3];
            for i in 0..self.dim_r(){
                pos[i]=atom.position()[i];
            }
            position.push(pos);
            atomtype.push(i32::from(atom.atom_type()));
        }
        let mut unit_cell=Cell::new(&lattice,&position,&atomtype);
        println!("{:?}",unit_cell);
        let dataset=Dataset::new(&mut unit_cell,1e-5);
        dataset
    }
}

#[cfg(test)]                                                                                                        mod tests {
    use crate::OrbProj;
    use ndarray::*;
    use crate::Model;
    use num_complex::Complex;
    use spglib::dataset::Dataset;
    use spglib::cell::Cell;


    #[test]
    fn symm_test(){
        let li:Complex<f64>=1.0*Complex::i();
        let t1=1.0+0.0*li;
        let t2=0.1+0.0*li;
        let t3=0.0+0.0*li;
        let delta=0.0;
        let dim_r:usize=3;
        let norb:usize=2;
        let lat=arr2(&[[1.0,0.0,0.0],[0.5,3.0_f64.sqrt()/2.0,0.0],[0.0,0.0,1.0]]);
        let orb=arr2(&[[1.0/3.0,1.0/3.0,0.0],[2.0/3.0,2.0/3.0,0.0]]);
        let mut model=Model::tb_model(dim_r,lat,orb,false,None);
        model.set_projection(&vec![OrbProj::pz,OrbProj::pz]);
        model.set_onsite(&arr1(&[delta,-delta]),0);
        model.add_hop(t1,0,1,&array![0,0,0],0);
        model.add_hop(t1,0,1,&array![-1,0,0],0);
        model.add_hop(t1,0,1,&array![0,-1,0],0);
        model.add_hop(t2,0,0,&array![1,0,0],0);
        model.add_hop(t2,1,1,&array![1,0,0],0);
        model.add_hop(t2,0,0,&array![0,1,0],0);
        model.add_hop(t2,1,1,&array![0,1,0],0);
        model.add_hop(t2,0,0,&array![1,-1,0],0);
        model.add_hop(t2,1,1,&array![1,-1,0],0);
        model.add_hop(t3,0,1,&array![1,-1,0],0);
        model.add_hop(t3,0,1,&array![-1,1,0],0);
        model.add_hop(t3,0,1,&array![-1,-1,0],0);
        let space_data=model.space_group();
        println!("{}",space_data.international_symbol);


        let lat=[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,2.0]];
        let atom_position=vec![[0.5,0.5,0.5]];
        let atom_type=vec![1,1];
        let mut unit_cell=Cell::new(&lat,&atom_position,&atom_type);
        let dataset=Dataset::new(&mut unit_cell,1e-5);
        println!("{}",dataset.international_symbol);
    }
}
