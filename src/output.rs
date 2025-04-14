//!这个模块是用来输出各种标准格式的, 包括
//!
//!wannier90_hr.dat 格式
//!
//!wannier90_centres.xyz 格式
//!
//!wannier90.win 格式
//!
//!整合的 wannier90 格式
//!
//!POSCAR 格式
use crate::basis::find_R;
use crate::basis::index_R;
use crate::{Model, comm, gen_kmesh};
use ndarray::concatenate;
use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::conjugate;
use ndarray_linalg::*;
use ndarray_linalg::{Eigh, UPLO};
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::ops::AddAssign;
use std::ops::MulAssign;

impl Model {
    pub fn output_hr(&self, path: &str, seedname: &str) {
        //! 这个函数是用来将 tight-binding 模型输出到 wannier90_hr.dat 格式的

        let n_R = self.hamR.nrows(); //length of hamR
        let mut hr_name = String::new();
        hr_name.push_str(path);
        hr_name.push_str(seedname);
        hr_name.push_str("_hr.dat");
        let mut file = File::create(hr_name).expect("Unable to BAND.dat");
        writeln!(file, "{}", self.nsta());
        writeln!(file, "{}", n_R);
        let mut weight = String::new();
        let lines = n_R.div_euclid(15);
        let last_lines = n_R % 15;
        if lines != 0 {
            for i in 0..lines {
                weight.push_str(
                    "    1    1    1    1    1    1    1    1    1    1    1    1    1    1    1\n",
                );
            }
        }
        for i in 0..last_lines {
            weight.push_str("    1");
        }
        writeln!(file, "{}", weight);
        //接下来我们进行数据的写入
        match self.dim_r {
            0 => {
                let mut s = String::new();
                let ham = self.ham.slice(s![0, .., ..]);
                for orb_1 in 0..self.nsta() {
                    for orb_2 in 0..self.nsta() {
                        s.push_str(&format!(
                            "0    0    0    {:15.8}    {:15.8}\n",
                            ham[[orb_1, orb_2]].re,
                            ham[[orb_1, orb_2]].im
                        ));
                    }
                }
            }
            1 => {
                let max_R1 = self.hamR.outer_iter().map(|x| x[[0]].abs()).max().unwrap();
                let mut s = String::new();
                for i in -max_R1..max_R1 {
                    let R_exist = find_R(&self.hamR, &array![i as isize]);
                    let R_inv_exist = find_R(&self.hamR, &(-array![i as isize]));
                    if R_exist {
                        let r0 = index_R(&self.hamR, &array![i as isize]);
                        let ham = self.ham.slice(s![r0, .., ..]);
                        for orb_1 in 0..self.nsta() {
                            for orb_2 in 0..self.nsta() {
                                s.push_str(&format!(
                                    "{:>3}    0    0    {:>3}    {:>3}    {:>15.8}    {:>15.8}\n",
                                    i,
                                    orb_1,
                                    orb_2,
                                    ham[[orb_1, orb_2]].re,
                                    ham[[orb_1, orb_2]].im
                                ));
                            }
                        }
                    } else if R_inv_exist {
                        let r0 = index_R(&self.hamR, &(-array![i as isize]));
                        let ham = self.ham.slice(s![r0, .., ..]);
                        for orb_1 in 0..self.nsta() {
                            for orb_2 in 0..self.nsta() {
                                s.push_str(&format!(
                                    "{:>3}    0    0    {:>3}    {:>3}    {:>15.8}    {:>15.8}\n",
                                    i,
                                    orb_1,
                                    orb_2,
                                    ham[[orb_1, orb_2]].re,
                                    -ham[[orb_1, orb_2]].im
                                ));
                            }
                        }
                    }
                }
                writeln!(file, "{}", s);
            }
            2 => {
                let max_values = self
                    .hamR
                    .fold_axis(Axis(0), isize::min_value(), |max, &value| {
                        *max.max(&value.abs())
                    });
                let mut s = String::new();
                for R1 in -max_values[[0]]..max_values[[0]] {
                    for R2 in -max_values[[1]]..max_values[[1]] {
                        let R_exist = find_R(&self.hamR, &array![R1 as isize, R2 as isize]);
                        let R_inv_exist = find_R(&self.hamR, &(-array![R1 as isize, R2 as isize]));
                        if R_exist {
                            let r0 = index_R(&self.hamR, &array![R1 as isize, R2 as isize]);
                            let ham = self.ham.slice(s![r0, .., ..]);
                            for orb_1 in 0..self.nsta() {
                                for orb_2 in 0..self.nsta() {
                                    s.push_str(&format!(
                                        "{:>3}  {:>3}    0    {:>3}    {:>3}    {:>15.8}    {:>15.8}\n",
                                        R1,
                                        R2,
                                        orb_1,
                                        orb_2,
                                        ham[[orb_1, orb_2]].re,
                                        ham[[orb_1, orb_2]].im
                                    ));
                                }
                            }
                        } else if R_inv_exist {
                            let r0 = index_R(&self.hamR, &(-array![R1 as isize, R2 as isize]));
                            let ham = self.ham.slice(s![r0, .., ..]);
                            for orb_1 in 0..self.nsta() {
                                for orb_2 in 0..self.nsta() {
                                    s.push_str(&format!(
                                        "{:>3}  {:>3}    0    {:>3}    {:>3}    {:>15.8}    {:>15.8}\n",
                                        R1,
                                        R2,
                                        orb_1,
                                        orb_2,
                                        ham[[orb_1, orb_2]].re,
                                        -ham[[orb_1, orb_2]].im
                                    ));
                                }
                            }
                        }
                    }
                }
                writeln!(file, "{}", s);
            }
            3 => {
                let max_values = self
                    .hamR
                    .fold_axis(Axis(0), isize::min_value(), |max, &value| {
                        *max.max(&value.abs())
                    });
                let mut s = String::new();
                for R1 in -max_values[[0]]..max_values[[0]] {
                    for R2 in -max_values[[1]]..max_values[[1]] {
                        for R3 in -max_values[[2]]..max_values[[2]] {
                            let R_exist =
                                find_R(&self.hamR, &array![R1 as isize, R2 as isize, R3 as isize]);
                            let R_inv_exist = find_R(
                                &self.hamR,
                                &(-array![R1 as isize, R2 as isize, R3 as isize]),
                            );
                            if R_exist {
                                let r0 = index_R(&self.hamR, &array![R1 as isize, R2 as isize]);
                                let ham = self.ham.slice(s![r0, .., ..]);
                                for orb_1 in 0..self.nsta() {
                                    for orb_2 in 0..self.nsta() {
                                        s.push_str(&format!(
                                            "{:>3}  {:>3}  {:>3}  {:>3}  {:>3}    {:>15.8}    {:>15.8}\n",
                                            R1,
                                            R2,
                                            R3,
                                            orb_1+1,
                                            orb_2+1,
                                            ham[[orb_1, orb_2]].re,
                                            ham[[orb_1, orb_2]].im
                                        ));
                                    }
                                }
                            } else if R_inv_exist {
                                let r0 = index_R(
                                    &self.hamR,
                                    &(-array![R1 as isize, R2 as isize, R3 as isize]),
                                );
                                let ham = self.ham.slice(s![r0, .., ..]);
                                for orb_1 in 0..self.nsta() {
                                    for orb_2 in 0..self.nsta() {
                                        s.push_str(&format!(
                                            "{:>3}  {:>3}  {:>3}  {:>3}  {:>3}    {:>15.8}    {:>15.8}\n",
                                            R1,
                                            R2,
                                            R3,
                                            orb_1+1,
                                            orb_2+1,
                                            ham[[orb_1, orb_2]].re,
                                           -ham[[orb_1, orb_2]].im
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
                writeln!(file, "{}", s);
            }
            _ => todo!(),
        }
    }

    pub fn output_POSCAR(&self, path: &str) {
        let mut name = String::new();
        name.push_str(path);
        name.push_str("POSCAR");
        let mut file = File::create(&name).expect("Unable to BAND.dat");
        writeln!(file, "Generate by Rustb");
        writeln!(file, "1.0");
        let s=match self.dim_r{
            3=>{
                let mut s = String::new();
                s.push_str(&format!("    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}",self.lat[[0,0]],self.lat[[0,1]],self.lat[[0,2]],self.lat[[1,0]],self.lat[[1,1]],self.lat[[1,2]],self.lat[[2,0]],self.lat[[2,1]],self.lat[[2,2]]));
                s
            },
            2=>{
                let mut s = String::new();
                s.push_str(&format!("    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}",self.lat[[0,0]],self.lat[[0,1]],0.0,self.lat[[1,0]],self.lat[[1,1]],0.0,0.0,0.0,10.0));
                s
            },
            1=>{
                let mut s = String::new();
                s.push_str(&format!("    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}",self.lat[[0,0]],0.0,0.0,0.0,10.0,0.0,0.0,0.0,10.0));
                s
            },
            _=>{
                panic!("Wrong! for POSCAR output, the dim_r of the model must be 1, 2 or 3, but yours {}",self.dim_r);
            }
        };
        writeln!(file,"{}",s);
        //开始弄atom
        let mut atom_type=vec![];
        let mut atom_num=vec![];
        let mut new_atom_position:Vec<Vec<Array1<f64>>>=Vec::new();
        for i in 0..self.natom(){
            let mut have_atom=false;
            for j in 0..atom_type.len(){
                if self.atoms[i].atom_type()==atom_type[j]{
                    have_atom=true;
                    atom_num[j]+=1;
                    new_atom_position[j].push(self.atom_position().row(i).to_owned());
                }
            }
            if have_atom==false{
                atom_num.push(1);
                atom_type.push(self.atoms[i].atom_type());
                new_atom_position.push(vec![self.atom_position().row(i).to_owned()]);
            }

        }
        let mut s=String::new();
        for i in 0..atom_type.len(){
            s.push_str(&format!("   {}",atom_type[i]));
        }
        writeln!(file,"{}",s);
        let mut s=String::new();
        for i in 0..atom_type.len(){
            s.push_str(&format!("{:>4}",atom_num[i]));
        }
        writeln!(file,"{}",s);
        writeln!(file, "Direct");
        let mut s = String::new();
        for i in 0..atom_type.len(){
            for j in 0..new_atom_position[i].len(){
                let s=match self.dim_r{
                    3=>{
                        let mut s=String::new();
                        s.push_str(&format!("{:>15.8}   {:>15.8}   {:>15.8}", new_atom_position[i][j][[0]],new_atom_position[i][j][[1]],new_atom_position[i][j][[2]]));
                        s
                    },
                    2=>{
                        let mut s=String::new();
                        s.push_str(&format!("{:>15.8}   {:>15.8}   {:>15.8}", new_atom_position[i][j][[0]],new_atom_position[i][j][[1]],0.0));
                        s
                    },
                    1=>{
                        let mut s=String::new();
                        s.push_str(&format!("{:>15.8}   {:>15.8}   {:>15.8}", new_atom_position[i][j][[0]],0.0,0.0));
                        s
                    },
                    _=>{
                        panic!("Wrong! for POSCAR output, the dim_r of the model must be 1, 2 or 3, but yours {}",self.dim_r);
                    }
                };
                writeln!(file,"{}",s);
            }
        }
    }

    pub fn output_win(&self, path: &str, seedname: &str) {
        //!这个是用来输出 win 文件的. 这里projection 需要人为添加, 因为没有保存相关的projection 数据
        let mut name = String::new();
        name.push_str(path);
        name.push_str(seedname);
        name.push_str(".win");
        let mut file = File::create(name).expect("Wrong, can't create seedname.win");
        writeln!(file, "begin atoms_cart");
        for at in self.atoms.iter() {
            let atom_position = at.position();
            match self.dim_r {
                3 => {
                    writeln!(
                        file,
                        "{}  {:>10.6}  {:>10.6}  {:>10.6}",
                        at.atom_type(),
                        atom_position[0],
                        atom_position[1],
                        atom_position[1]
                    );
                }
                2 => {
                    writeln!(
                        file,
                        "{}  {:>10.6}  {:>10.6}  {:>10.6}",
                        at.atom_type(),
                        atom_position[0],
                        atom_position[1],
                        0.0
                    );
                }
                1 => {
                    writeln!(
                        file,
                        "{}  {:>10.6}  {:>10.6}  {:>10.6}",
                        at.atom_type(),
                        atom_position[0],
                        0.0,
                        0.0
                    );
                }
                _ => panic!("Wrong, your model's dim_r is not 1,2 or 3"),
            }
        }
        writeln!(file, "end atoms_cart");
        writeln!(file, "\n");
        writeln!(file, "begin unit_cell_cart");
        match self.dim_r {
            3 => {
                let mut s = String::new();
                for i in 0..3 {
                    for j in 0..3 {
                        s.push_str(&format!("{:>10.6}  ", self.lat[[i, j]]));
                    }
                    writeln!(file, "{}", s);
                }
            }
            2 => {
                let mut s = String::new();
                for i in 0..2 {
                    for j in 0..2 {
                        s.push_str(&format!("{:>10.6}  ", self.lat[[i, j]]));
                    }
                    s.push_str("   0.000000");
                    writeln!(file, "{}", s);
                }
                writeln!(file, "   0.000000     0.000000     1.000000");
            }
            1 => {
                let mut s = String::new();
                s.push_str(&format!("{:>10.6}  ", self.lat[[0, 0]]));
                s.push_str("   0.000000     0.000000");
                writeln!(file, "{}", s);
                writeln!(file, "   0.000000     0.000000     1.000000");
                writeln!(file, "   0.000000     0.000000     1.000000");
            }
            _ => {
                panic!(
                    "Wrong! Using output win file, the dim_r of model mut be 1, 2, or 3, but yours {}",
                    self.dim_r
                )
            }
        }
        writeln!(file, "end unit_cell_cart");
        writeln!(file, "\n");
        //还差投影轨道
    }
}
