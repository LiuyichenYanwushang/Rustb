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
use crate::Model;
use crate::math::comm;
use crate::kpoints::gen_kmesh;
use crate::error::{TbError, Result};
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
                for orb_2 in 0..self.nsta() {
                    for orb_1 in 0..self.nsta() {
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
                    match (find_R(&self.hamR, &array![i as isize]),find_R(&self.hamR, &(-array![i as isize]))){
                        (Some(r0),_)=>{
                            let ham = self.ham.slice(s![r0, .., ..]);
                            for orb_2 in 0..self.nsta() {
                                for orb_1 in 0..self.nsta() {
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
                        },
                        (None,Some(r0))=>{
                            let ham = self.ham.slice(s![r0, .., ..]);
                            for orb_2 in 0..self.nsta() {
                                for orb_1 in 0..self.nsta() {
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
                        },
                        (None,None)=>{},
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
                        let R0=array![R1 as isize,R2 as isize];
                        let R0_inv=-array![R1 as isize,R2 as isize];
                        match (find_R(&self.hamR, &R0),find_R(&self.hamR, &R0_inv)){
                            (Some(r0),_)=>{
                                let ham = self.ham.slice(s![r0, .., ..]);
                                for orb_2 in 0..self.nsta() {
                                    for orb_1 in 0..self.nsta() {
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
                            },
                            (None,Some(r0))=>{
                                let ham = self.ham.slice(s![r0, .., ..]);
                                for orb_2 in 0..self.nsta() {
                                    for orb_1 in 0..self.nsta() {
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
                            },
                            (None,None)=>{},
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
                            let R0=array![R1 as isize,R2 as isize,R3 as isize];
                            let R0_inv=-array![R1 as isize,R2 as isize,R3 as isize];
                            match (find_R(&self.hamR, &R0),find_R(&self.hamR, &R0_inv)){
                                (Some(r0),_)=>{
                                    let ham = self.ham.slice(s![r0, .., ..]);
                                    for orb_2 in 0..self.nsta() {
                                        for orb_1 in 0..self.nsta() {
                                            s.push_str(&format!(
                                                "{:>3}  {:>3}  {:>3}    {:>3}    {:>3}    {:>15.8}    {:>15.8}\n",
                                                R1,
                                                R2,
                                                R3,
                                                orb_1,
                                                orb_2,
                                                ham[[orb_1, orb_2]].re,
                                                ham[[orb_1, orb_2]].im
                                            ));
                                        }
                                    }
                                },
                                (None,Some(r0))=>{
                                    let ham = self.ham.slice(s![r0, .., ..]);
                                    for orb_2 in 0..self.nsta() {
                                        for orb_1 in 0..self.nsta() {
                                            s.push_str(&format!(
                                                "{:>3}  {:>3}  {:>3}    {:>3}    {:>3}    {:>15.8}    {:>15.8}\n",
                                                R1,
                                                R2,
                                                R3,
                                                orb_1,
                                                orb_2,
                                                ham[[orb_1, orb_2]].re,
                                                -ham[[orb_1, orb_2]].im
                                            ));
                                        }
                                    }
                                },
                                (None,None)=>{},
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
        let s = match self.dim_r {
            3 => {
                let mut s = String::new();
                s.push_str(&format!("    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}",self.lat[[0,0]],self.lat[[0,1]],self.lat[[0,2]],self.lat[[1,0]],self.lat[[1,1]],self.lat[[1,2]],self.lat[[2,0]],self.lat[[2,1]],self.lat[[2,2]]));
                s
            }
            2 => {
                let mut s = String::new();
                s.push_str(&format!("    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}",self.lat[[0,0]],self.lat[[0,1]],0.0,self.lat[[1,0]],self.lat[[1,1]],0.0,0.0,0.0,10.0));
                s
            }
            1 => {
                let mut s = String::new();
                s.push_str(&format!("    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}",self.lat[[0,0]],0.0,0.0,0.0,10.0,0.0,0.0,0.0,10.0));
                s
            }
            _ => {
                panic!(
                    "Wrong! for POSCAR output, the dim_r of the model must be 1, 2 or 3, but yours {}",
                    self.dim_r
                );
            }
        };
        writeln!(file, "{}", s);
        //开始弄atom
        let mut atom_type = vec![];
        let mut atom_num = vec![];
        let mut new_atom_position: Vec<Vec<Array1<f64>>> = Vec::new();
        for i in 0..self.natom() {
            let mut have_atom = false;
            for j in 0..atom_type.len() {
                if self.atoms[i].atom_type() == atom_type[j] {
                    have_atom = true;
                    atom_num[j] += 1;
                    new_atom_position[j].push(self.atom_position().row(i).to_owned());
                }
            }
            if have_atom == false {
                atom_num.push(1);
                atom_type.push(self.atoms[i].atom_type());
                new_atom_position.push(vec![self.atom_position().row(i).to_owned()]);
            }
        }
        let mut s = String::new();
        for i in 0..atom_type.len() {
            s.push_str(&format!("   {}", atom_type[i]));
        }
        writeln!(file, "{}", s);
        let mut s = String::new();
        for i in 0..atom_type.len() {
            s.push_str(&format!("{:>4}", atom_num[i]));
        }
        writeln!(file, "{}", s);
        writeln!(file, "Direct");
        let mut s = String::new();
        for i in 0..atom_type.len() {
            for j in 0..new_atom_position[i].len() {
                let s = match self.dim_r {
                    3 => {
                        let mut s = String::new();
                        s.push_str(&format!(
                            "{:>15.8}   {:>15.8}   {:>15.8}",
                            new_atom_position[i][j][[0]],
                            new_atom_position[i][j][[1]],
                            new_atom_position[i][j][[2]]
                        ));
                        s
                    }
                    2 => {
                        let mut s = String::new();
                        s.push_str(&format!(
                            "{:>15.8}   {:>15.8}   {:>15.8}",
                            new_atom_position[i][j][[0]],
                            new_atom_position[i][j][[1]],
                            0.0
                        ));
                        s
                    }
                    1 => {
                        let mut s = String::new();
                        s.push_str(&format!(
                            "{:>15.8}   {:>15.8}   {:>15.8}",
                            new_atom_position[i][j][[0]],
                            0.0,
                            0.0
                        ));
                        s
                    }
                    _ => {
                        panic!(
                            "Wrong! for POSCAR output, the dim_r of the model must be 1, 2 or 3, but yours {}",
                            self.dim_r
                        );
                    }
                };
                writeln!(file, "{}", s);
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
        writeln!(file, "begin projections");
        writeln!(file, "end projections");
    }
    pub fn output_xyz(&self, path: &str, seedname: &str) {
        //!这个是用来输出 xyz 文件的. 这里projection 需要人为添加, 因为没有保存相关的projection 数据
        let mut name = String::new();
        name.push_str(path);
        name.push_str(seedname);
        name.push_str("_centres.xyz");
        let mut file = File::create(name).expect("Wrong, can't create seedname.win");
        let number = self.nsta() + self.natom();
        let orb_real = self.orb.dot(&self.lat);
        let atom_position_real = self.atom_position().dot(&self.lat);
        writeln!(file, "{}", number);
        writeln!(file, "Wannier centres, written by Rustb");
        let mut s = match self.dim_r {
            3 => {
                let mut s = String::new();
                for i in 0..self.norb() {
                    s.push_str(&format!(
                        "X{:>20.8}{:>17.8}{:>17.8}\n",
                        orb_real[[i, 0]],
                        orb_real[[i, 1]],
                        orb_real[[i, 2]]
                    ));
                }
                if self.spin {
                    for i in 0..self.norb() {
                        s.push_str(&format!(
                            "X{:>20.8}{:>17.8}{:>17.8}\n",
                            orb_real[[i, 0]],
                            orb_real[[i, 1]],
                            orb_real[[i, 2]]
                        ));
                    }
                }
                for i in 0..self.natom() - 1 {
                    s.push_str(&format!(
                        "{}{:>19.8}{:>17.8}{:>17.8}\n",
                        self.atoms[i].atom_type(),
                        atom_position_real[[i, 0]],
                        atom_position_real[[i, 1]],
                        atom_position_real[[i, 2]]
                    ));
                }
                let i = self.natom() - 1;
                s.push_str(&format!(
                    "{}{:>19.8}{:>17.8}{:>17.8}",
                    self.atoms[i].atom_type(),
                    atom_position_real[[i, 0]],
                    atom_position_real[[i, 1]],
                    atom_position_real[[i, 2]]
                ));
                s
            }
            2 => {
                let mut s = String::new();
                for i in 0..self.norb() {
                    s.push_str(&format!(
                        "X{:>20.8}{:>17.8}       0.00000000\n",
                        orb_real[[i, 0]],
                        orb_real[[i, 1]]
                    ));
                }
                if self.spin {
                    for i in 0..self.norb() {
                        s.push_str(&format!(
                            "X{:>20.8}{:>17.8}       0.00000000\n",
                            orb_real[[i, 0]],
                            orb_real[[i, 1]]
                        ));
                    }
                }
                for i in 0..self.natom() - 1 {
                    s.push_str(&format!(
                        "{}{:>19.8}{:>17.8}       0.00000000\n",
                        self.atoms[i].atom_type(),
                        atom_position_real[[i, 0]],
                        atom_position_real[[i, 1]]
                    ));
                }
                let i = self.natom() - 1;
                s.push_str(&format!(
                    "{}{:>19.8}{:>17.8}       0.00000000",
                    self.atoms[i].atom_type(),
                    atom_position_real[[i, 0]],
                    atom_position_real[[i, 1]]
                ));
                s
            }
            1 => {
                let mut s = String::new();
                for i in 0..self.norb() {
                    s.push_str(&format!(
                        "X{:>20.8}       0.00000000       0.00000000\n",
                        orb_real[[i, 0]]
                    ));
                }
                if self.spin {
                    for i in 0..self.norb() {
                        s.push_str(&format!(
                            "X{:>20.8}       0.00000000       0.00000000\n",
                            orb_real[[i, 0]]
                        ));
                    }
                }
                for i in 0..self.natom() - 1 {
                    s.push_str(&format!(
                        "{}{:>19.8}       0.00000000       0.00000000\n",
                        self.atoms[i].atom_type(),
                        atom_position_real[[i, 0]]
                    ));
                }
                let i = self.natom() - 1;
                s.push_str(&format!(
                    "{}{:>19.8}       0.00000000       0.00000000",
                    self.atoms[i].atom_type(),
                    atom_position_real[[i, 0]]
                ));
                s
            }
            _ => {
                panic!(
                    "Wrong!, the dim_r must be 1,2 or 3, but yours {}",
                    self.dim_r
                );
            }
        };
        writeln!(file, "{}", s);
    }

    ///这个函数是用来快速画能带图的, 用python画图, 因为Rust画图不太方便.
    #[allow(non_snake_case)]
    pub fn show_band(
        &self,
        path: &Array2<f64>,
        label: &Vec<&str>,
        nk: usize,
        name: &str,
    ) -> Result<()> {
        use gnuplot::AutoOption::*;
        use gnuplot::AxesCommon;
        use gnuplot::Tick::*;
        use gnuplot::{Caption, Color, Figure, Font, LineStyle, Solid};
        use std::fs::create_dir_all;
        use std::path::Path;
        if path.len_of(Axis(0)) != label.len() {
            panic!(
                "Error, the path's length {} and label's length {} must be equal!",
                path.len_of(Axis(0)),
                label.len()
            )
        }
        let (k_vec, k_dist, k_node) = self.k_path(&path, nk)?;
        let eval = self.solve_band_all_parallel(&k_vec);
        create_dir_all(name).map_err(|e| TbError::DirectoryCreation {
            path: name.to_string(),
            message: e.to_string(),
        })?;
        let mut name0 = String::new();
        name0.push_str("./");
        name0.push_str(&name);
        let name = name0;
        let mut band_name = name.clone();
        band_name.push_str("/BAND.dat");
        let band_name = Path::new(&band_name);
        let mut file = File::create(band_name).expect("Unable to BAND.dat");
        for i in 0..nk {
            let mut s = String::new();
            let aa = format!("{:.6}", k_dist[[i]]);
            s.push_str(&aa);
            for j in 0..self.nsta() {
                if eval[[i, j]] >= 0.0 {
                    s.push_str("     ");
                } else {
                    s.push_str("    ");
                }
                let aa = format!("{:.6}", eval[[i, j]]);
                s.push_str(&aa);
            }
            writeln!(file, "{}", s)?;
        }
        let mut k_name = name.clone();
        k_name.push_str("/KLABELS");
        let k_name = Path::new(&k_name);
        let mut file = File::create(k_name).expect("Unable to create KLBAELS"); //写下高对称点的位置
        for i in 0..path.len_of(Axis(0)) {
            let mut s = String::new();
            let aa = format!("{:.6}", k_node[[i]]);
            s.push_str(&aa);
            s.push_str("      ");
            s.push_str(&label[i]);
            writeln!(file, "{}", s)?;
        }
        let mut py_name = name.clone();
        py_name.push_str("/print.py");
        let py_name = Path::new(&py_name);
        let mut file = File::create(py_name).expect("Unable to create print.py");
        writeln!(
            file,
            "import numpy as np\nimport matplotlib.pyplot as plt\ndata=np.loadtxt('BAND.dat')\nk_nodes=[]\nlabel=[]\nf=open('KLABELS')\nfor i in f.readlines():\n    k_nodes.append(float(i.split()[0]))\n    label.append(i.split()[1])\nfig,ax=plt.subplots()\nax.plot(data[:,0],data[:,1:],c='b')\nfor x in k_nodes:\n    ax.axvline(x,c='k')\nax.set_xticks(k_nodes)\nax.set_xticklabels(label)\nax.set_xlim([0,k_nodes[-1]])\nfig.savefig('band.pdf')"
        );
        //开始绘制pdf图片
        let mut fg = Figure::new();
        let x: Vec<f64> = k_dist.to_vec();
        let axes = fg.axes2d();
        for i in 0..self.nsta() {
            let y: Vec<f64> = eval.slice(s![.., i]).to_owned().to_vec();
            axes.lines(&x, &y, &[Color("black"), LineStyle(Solid)]);
        }
        let axes = axes.set_x_range(Fix(0.0), Fix(k_node[[k_node.len() - 1]]));
        let label = label.clone();
        let mut show_ticks = Vec::new();
        for i in 0..k_node.len() {
            let A = k_node[[i]];
            let B = label[i];
            show_ticks.push(Major(A, Fix(B)));
        }
        axes.set_x_ticks_custom(
            show_ticks.into_iter(),
            &[],
            &[Font("Times New Roman", 24.0)],
        );

        let k_node = k_node.to_vec();
        let mut pdf_name = name.clone();
        pdf_name.push_str("/plot.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
        Ok(())
    }
}


pub fn draw_heatmap<A: Data<Elem = f64>>(data: &ArrayBase<A, Ix2>, name: &str) {
    //!这个函数是用来画热图的, 给定一个二维矩阵, 会输出一个像素图片
    use gnuplot::{AutoOption::Fix, AxesCommon, Figure, HOT, RAINBOW};
    let mut fg = Figure::new();
    let (height, width): (usize, usize) = (data.shape()[0], data.shape()[1]);
    let mut heatmap_data = vec![];

    for j in 0..width {
        for i in 0..height {
            heatmap_data.push(data[(i, j)]);
        }
    }
    let axes = fg.axes2d();
    axes.set_title("Heatmap", &[]);
    axes.set_cb_label("Values", &[]);
    axes.set_palette(RAINBOW);
    axes.image(heatmap_data.iter(), width, height, None, &[]);
    let size = data.shape();
    let axes = axes.set_x_range(Fix(0.0), Fix((size[0] - 1) as f64));
    let axes = axes.set_y_range(Fix(0.0), Fix((size[1] - 1) as f64));
    let axes = axes.set_aspect_ratio(Fix(1.0));
    fg.set_terminal("pdfcairo", name);
    fg.show().expect("Unable to draw heatmap");
}
