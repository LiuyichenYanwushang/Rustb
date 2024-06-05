#[derive(Clone,Copy)]
enum orb_projection{
    s,
    px,
    py,
    pz,
    dxy,
    dyz,
    dxz,
    dz2,
    dx2y2,
}
#[derive(Clone,Copy)]
enum atom_type{
    H,
    He,
    Li,
    Be,
    B,
    C,
    N,
    O,
    F,
    Ne,
    Na,
    Mg,
    Al,
    Si,
    P,
    S,
    Cl,
    Ar,
    K,
    Ca,
    Se,
    Ti,
    V,
    Cr,
    Mn,
    Fe,
    Co,
    Ni,
    Cu,
    Zn,
    Ga,
    Ge,
    As,
    Se,
    Br,
    Kr,
    Rb,
    Sr,
    Y,
    Zr,
    Nb,
    Mo,
    Tc,
    Ru,
    Rh,
    Pd,
    Ag,
    Cd,
    In,
    Sn,
    Sb,
    Te,
    I,
    Xe,
    Cs,
    Ba,
    La,
    Ce,
    Pr,
    Nd,
    Pm,
    Sm,
    Eu,
    Gd,
    Tb,
    Dy,
    Ho,
    Er,
    Tm,
    Yb,
    Lu,
    Hf,
    Ta,
    W,
    Re,
    Os,
    Ir,
    Pt,
    Au,
    Hg,
    Tl,
    Pb,
    Bi,
    Po,
    At,
    Rn,
    Fr,
    Ra
}


#[derive(Clone)]
struct orb{
    position:Array1<f64>,
    projection:orb_projection,
}

impl orb{

    fn position(&self)->Array1<f64>{
        self.position
    }
    fn projection(&self)->orb_projection{
        self.projection
    }
    fn gen_orb(position:Array1<f64>,projection:orb_projection)->orb{
        orb{position,projection}
    }
}

#[derive(Clone)]
struct atom{
    position:Array1<f64>,
    orb:Vec<orb>,
    name:atom_type,
}

impl atom{
    fn position(&self)->Array1<f64>{
        self.position
    }
    fn push_orb(&self,orb:orb){
        self.orb.push(orb);
    }
    fn norb(&self)->usize{
        self.orb.len();
    }
    fn gen_atom(position:Array1<f64>,orb:&Vec<orb>,name:atom_type)->atom{
        atom{position,orb:orb.clone(),name}
    }
}
