use crate::atom_struct::{Atom, AtomType, OrbProj};
use crate::error::{TbError, Result};
use crate::{Gauge, Model, SpinDirection};
use crate::math::comm;
use crate::basis::find_R;
use ndarray::prelude::*;
use ndarray_linalg::conjugate;
use ndarray_linalg::*;
use num_complex::{Complex, Complex64};


impl Model{
    #[allow(non_snake_case)]
    pub fn from_hr(path: &str, file_name: &str, zero_energy: f64) -> Result<Model> {
        //! 这个函数是从 wannier90 中读取 TB 文件的
        //!
        //! 这里 path 表示 文件的位置, 可以使用绝对路径, 即 "/" 开头的路径, 也可以使用相对路径, 即运行 cargo run 时
        //! 候的文件夹作为起始路径.
        //!
        //! file_name 就是 wannier90 中的 seedname, 文件可以读取
        //! seedname.win, seedname_centres.xyz, seedname_hr.dat, 以及可选的 seedname_r.dat.
        //!
        //! 这里 seedname_centres.xyz 需要在 wannier90 中设置 write_xyz=true, 而
        //! seedname_hr.dat 需要设置 write_hr=true.
        //! 如果想要计算输运性质, 则需要 write_rmn=true, 能够给出一个seedname_r.dat.
        //!
        //! 此外, 对于高版本的wannier90, 如果想要保持较好的对称性, 建议将wannier90_wsvec.dat
        //! 也给出, 这样能够得到较好的对称结果

        use std::fs::File;
        use std::io::BufRead;
        use std::io::BufReader;
        use std::path::Path;

        let mut file_path = path.to_string();
        file_path.push_str(file_name);
        let mut hr_path = file_path.clone();
        hr_path.push_str("_hr.dat");

        let path = Path::new(&hr_path);
        let hr = File::open(path).map_err(|e| TbError::FileCreation {
            path: hr_path.clone(),
            message: format!("Unable to open HR file: {}", e),
        })?;
        let reader = BufReader::new(hr);
        let mut reads: Vec<String> = Vec::new();

        // 读取文件行
        for line in reader.lines() {
            let line = line.map_err(|e| TbError::Io(e))?;
            reads.push(line.clone());
        }

        // 获取轨道数和R点数
        let nsta = reads[1].trim().parse::<usize>()
            .map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse nsta: {}", e) })?;
        let n_R = reads[2].trim().parse::<usize>()
            .map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse n_R: {}", e) })?;
        let mut weights: Vec<usize> = Vec::new();
        let mut n_line: usize = 0;

        // 解析文件数据以获取权重
        for i in 3..reads.len() {
            if reads[i].contains(".") {
                n_line = i;
                break;
            }
            let string = reads[i].trim().split_whitespace();
            let string: Vec<_> = string.map(|x| x.parse::<usize>()
                .map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse weight: {}", e) }))
                .collect::<Result<Vec<_>>>()?;
            weights.extend(string.clone());
        }

        // 初始化哈密顿量矩阵
        let mut hamR = Array2::<isize>::zeros((1, 3));
        let mut ham = Array3::<Complex<f64>>::zeros((1, nsta, nsta));

        // 遍历每个R点并填充哈密顿量
        for i in 0..n_R {
            let mut string = reads[i * nsta * nsta + n_line].trim().split_whitespace();
            let a = string.next().ok_or_else(|| TbError::FileParse { file: hr_path.clone(), message: "Missing R vector component".to_string() })?
                .parse::<isize>().map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
            let b = string.next().ok_or_else(|| TbError::FileParse { file: hr_path.clone(), message: "Missing R vector component".to_string() })?
                .parse::<isize>().map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
            let c = string.next().ok_or_else(|| TbError::FileParse { file: hr_path.clone(), message: "Missing R vector component".to_string() })?
                .parse::<isize>().map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;

            if a == 0 && b == 0 && c == 0 {
                for ind_i in 0..nsta {
                    for ind_j in 0..nsta {
                        let mut string = reads[i * nsta * nsta + ind_i * nsta + ind_j + n_line]
                            .trim()
                            .split_whitespace();
                        let re = string.nth(5).ok_or_else(|| TbError::FileParse { file: hr_path.clone(), message: "Missing Hamiltonian real part".to_string() })?
                            .parse::<f64>().map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse Hamiltonian real part: {}", e) })?;
                        let im = string.next().ok_or_else(|| TbError::FileParse { file: hr_path.clone(), message: "Missing Hamiltonian imaginary part".to_string() })?
                            .parse::<f64>().map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse Hamiltonian imaginary part: {}", e) })?;
                        ham[[0, ind_j, ind_i]] = Complex::new(re, im) / (weights[i] as f64);
                    }
                }
            } else {
                let mut matrix = Array3::<Complex<f64>>::zeros((1, nsta, nsta));
                for ind_i in 0..nsta {
                    for ind_j in 0..nsta {
                        let mut string = reads[i * nsta * nsta + ind_i * nsta + ind_j + n_line]
                            .trim()
                            .split_whitespace();
                        let re = string.nth(5).ok_or_else(|| TbError::FileParse { file: hr_path.clone(), message: "Missing Hamiltonian real part".to_string() })?
                            .parse::<f64>().map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse Hamiltonian real part: {}", e) })?;
                        let im = string.next().ok_or_else(|| TbError::FileParse { file: hr_path.clone(), message: "Missing Hamiltonian imaginary part".to_string() })?
                            .parse::<f64>().map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse Hamiltonian imaginary part: {}", e) })?;
                        matrix[[0, ind_j, ind_i]] = Complex::new(re, im) / (weights[i] as f64);
                        // wannier90 里面是按照纵向排列的矩阵
                    }
                }
                ham.append(Axis(0), matrix.view())
                    .map_err(|e| TbError::Linalg(ndarray_linalg::error::LinalgError::Shape(e)))?;
                hamR.append(Axis(0), arr2(&[[a, b, c]]).view())
                    .map_err(|e| TbError::Linalg(ndarray_linalg::error::LinalgError::Shape(e)))?;
            }
        }

        // 调整哈密顿量以匹配能量零点
        for i in 0..nsta {
            ham[[0, i, i]] -= Complex::new(zero_energy, 0.0);
        }
        //开始读取 .win 文件
        let mut reads: Vec<String> = Vec::new();
        let mut win_path = file_path.clone();
        win_path.push_str(".win"); //文件的位置
        let path = Path::new(&win_path); //转化为路径格式
        let hr = File::open(path).map_err(|e| TbError::FileCreation {
            path: win_path.clone(),
            message: format!("Unable to open win file: {}", e),
        })?;
        let reader = BufReader::new(hr);
        let mut reads: Vec<String> = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| TbError::Io(e))?;
            reads.push(line.clone());
        }
        let mut read_iter = reads.iter();
        let mut lat = Array2::<f64>::zeros((3, 3)); //晶格轨道坐标初始化
        let mut spin: bool = false; //体系自旋初始化
        let mut natom: usize = 0; //原子位置初始化
        let mut atom = Vec::new(); //原子位置坐标初始化
        let mut orb_proj = Vec::new();
        let mut proj_name = Vec::new();
        let mut proj_list: Vec<usize> = Vec::new();
        let mut atom_list: Vec<usize> = Vec::new();
        let mut atom_name: Vec<&str> = Vec::new();
        let mut atom_pos = Array2::<f64>::zeros((0, 3));
        let mut atom_proj = Vec::new();
        let mut norb = 0;
        loop {
            let a = read_iter.next();
            if a == None {
                break;
            } else {
                let a = a.ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Unexpected end of file".to_string() })?;
                if a.contains("begin unit_cell_cart") {
                    let mut lat1 = read_iter.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Missing lattice vector line".to_string() })?.trim().split_whitespace(); //将数字放到
                    let mut lat2 = read_iter.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Missing lattice vector line".to_string() })?.trim().split_whitespace();
                    let mut lat3 = read_iter.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Missing lattice vector line".to_string() })?.trim().split_whitespace();
                    for i in 0..3 {
                        lat[[0, i]] = lat1.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Missing lattice vector component".to_string() })?
                            .parse::<f64>().map_err(|e| TbError::FileParse { file: win_path.clone(), message: format!("Failed to parse lattice vector: {}", e) })?;
                        lat[[1, i]] = lat2.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Missing lattice vector component".to_string() })?
                            .parse::<f64>().map_err(|e| TbError::FileParse { file: win_path.clone(), message: format!("Failed to parse lattice vector: {}", e) })?;
                        lat[[2, i]] = lat3.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Missing lattice vector component".to_string() })?
                            .parse::<f64>().map_err(|e| TbError::FileParse { file: win_path.clone(), message: format!("Failed to parse lattice vector: {}", e) })?;
                    }
                } else if a.contains("spinors") && (a.contains("T") || a.contains("t")) {
                    spin = true;
                } else if a.contains("begin projections") {
                    loop {
                        let string = read_iter.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Unexpected end of file".to_string() })?;
                        if string.contains("end projections") {
                            break;
                        } else {
                            let prj: Vec<&str> = string
                                .split(|c| c == ',' || c == ';' || c == ':')
                                .map(|x| x.trim())
                                .collect();
                            let mut atom_orb_number: usize = 0;
                            let mut proj_orb = Vec::new();
                            for item in prj[1..].iter() {
                                let (aa, use_proj_orb): (usize, Vec<_>) = match (*item).trim() {
                                    "s" => (1, vec![OrbProj::s]),
                                    "p" => (3, vec![OrbProj::pz, OrbProj::px, OrbProj::py]),
                                    "d" => (
                                        5,
                                        vec![
                                            OrbProj::dz2,
                                            OrbProj::dxz,
                                            OrbProj::dyz,
                                            OrbProj::dx2y2,
                                            OrbProj::dxy,
                                        ],
                                    ),
                                    "f" => (
                                        7,
                                        vec![
                                            OrbProj::fz3,
                                            OrbProj::fxz2,
                                            OrbProj::fyz2,
                                            OrbProj::fzx2y2,
                                            OrbProj::fxyz,
                                            OrbProj::fxx23y2,
                                            OrbProj::fy3x2y2,
                                        ],
                                    ),
                                    "sp3" => (
                                        4,
                                        vec![
                                            OrbProj::sp3_1,
                                            OrbProj::sp3_2,
                                            OrbProj::sp3_3,
                                            OrbProj::sp3_4,
                                        ],
                                    ),
                                    "sp2" => {
                                        (3, vec![OrbProj::sp2_1, OrbProj::sp2_2, OrbProj::sp2_3])
                                    }
                                    "sp" => (2, vec![OrbProj::sp_1, OrbProj::sp_2]),
                                    "sp3d" => (
                                        5,
                                        vec![
                                            OrbProj::sp3d_1,
                                            OrbProj::sp3d_2,
                                            OrbProj::sp3d_3,
                                            OrbProj::sp3d_4,
                                            OrbProj::sp3d_5,
                                        ],
                                    ),
                                    "sp3d2" => (
                                        6,
                                        vec![
                                            OrbProj::sp3d2_1,
                                            OrbProj::sp3d2_2,
                                            OrbProj::sp3d2_3,
                                            OrbProj::sp3d2_4,
                                            OrbProj::sp3d2_5,
                                            OrbProj::sp3d2_6,
                                        ],
                                    ),
                                    "px" => (1, vec![OrbProj::px]),
                                    "py" => (1, vec![OrbProj::py]),
                                    "pz" => (1, vec![OrbProj::pz]),
                                    "dxy" => (1, vec![OrbProj::dxy]),
                                    "dxz" => (1, vec![OrbProj::dxz]),
                                    "dyz" => (1, vec![OrbProj::dyz]),
                                    "dz2" => (1, vec![OrbProj::dz2]),
                                    "dx2-y2" => (1, vec![OrbProj::dx2y2]),
                                    &_ => return Err(TbError::InvalidOrbitalProjection(
                                        format!("Unrecognized projection '{}' in seedname.win", item)
                                    )),
                                };
                                atom_orb_number += aa;
                                proj_orb.extend(use_proj_orb);
                            }
                            proj_list.push(atom_orb_number);
                            atom_proj.push(proj_orb);
                            let proj_type = AtomType::from_str(prj[0]);
                            proj_name.push(proj_type);
                        }
                    }
                } else if a.contains("begin atoms_cart") {
                    loop {
                        let string = read_iter.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Unexpected end of file".to_string() })?;
                        if string.contains("end atoms_cart") {
                            break;
                        } else {
                            let prj: Vec<&str> = string.split_whitespace().collect();
                            atom_name.push(prj[0]);
                            let a1 = prj[1].parse::<f64>()
                                .map_err(|e| TbError::FileParse { file: win_path.clone(), message: format!("Failed to parse atom position: {}", e) })?;
                            let a2 = prj[2].parse::<f64>()
                                .map_err(|e| TbError::FileParse { file: win_path.clone(), message: format!("Failed to parse atom position: {}", e) })?;
                            let a3 = prj[3].parse::<f64>()
                                .map_err(|e| TbError::FileParse { file: win_path.clone(), message: format!("Failed to parse atom position: {}", e) })?;
                            let a = array![a1, a2, a3];
                            atom_pos.push_row(a.view()); //这里我们不用win 里面的, 因为这个和orb没法对应, 如果没有xyz文件才考虑用这个
                        }
                    }
                }
            }
        }
        //开始读取 seedname_centres.xyz 文件
        let mut reads: Vec<String> = Vec::new();
        let mut xyz_path = file_path.clone();
        xyz_path.push_str("_centres.xyz");
        let path = Path::new(&xyz_path);
        let hr = File::open(path);
        let orb = if let Ok(hr) = hr {
            let reader = BufReader::new(hr);
            let mut reads: Vec<String> = Vec::new();
            for line in reader.lines() {
                let line = line.map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to read line: {}", e) })?;
                reads.push(line.clone());
            }
            //let nsta=reads[0].trim().parse::<usize>().unwrap()-natom;
            let norb = if spin { nsta / 2 } else { nsta };
            let mut orb = Array2::<f64>::zeros((norb, 3));
            for i in 0..norb {
                let a: Vec<&str> = reads[i + 2].trim().split_whitespace().collect();
                orb[[i, 0]] = a[1].parse::<f64>()
                    .map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to parse orbital position: {}", e) })?;
                orb[[i, 1]] = a[2].parse::<f64>()
                    .map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to parse orbital position: {}", e) })?;
                orb[[i, 2]] = a[3].parse::<f64>()
                    .map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to parse orbital position: {}", e) })?
            }
            orb = orb.dot(&lat.inv().map_err(TbError::Linalg)?);
            let mut new_atom_pos = Array2::<f64>::zeros((reads.len() - 2 - nsta, 3));
            let mut new_atom_name = Vec::new();
            for i in 0..reads.len() - 2 - nsta {
                let a: Vec<&str> = reads[i + 2 + nsta].trim().split_whitespace().collect();
                new_atom_pos[[i, 0]] = a[1].parse::<f64>()
                    .map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to parse atom position: {}", e) })?;
                new_atom_pos[[i, 1]] = a[2].parse::<f64>()
                    .map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to parse atom position: {}", e) })?;
                new_atom_pos[[i, 2]] = a[3].parse::<f64>()
                    .map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to parse atom position: {}", e) })?;
                new_atom_name.push(AtomType::from_str(a[0]));
            }
            //接下来如果wannier90.win 和 .xyz 文件的原子顺序不一致, 那么我们以xyz的原子顺序为准, 调整 atom_list

            for (i, name) in new_atom_name.iter().enumerate() {
                for (j, j_name) in proj_name.iter().enumerate() {
                    if name.as_ref().ok() == j_name.as_ref().ok() && name.is_ok() && j_name.is_ok() {
                        let use_pos = new_atom_pos.row(i).dot(&lat.inv().map_err(TbError::Linalg)?);
                        let use_atom = Atom::new(use_pos, proj_list[j], name.as_ref().unwrap().clone());
                        atom.push(use_atom);
                        orb_proj.extend(atom_proj[j].clone());
                    }
                }
            }
            orb
        } else {
            let mut orb = Array2::<f64>::zeros((0, 3));
            let atom_pos = atom_pos.dot(&lat.inv().map_err(TbError::Linalg)?);
            for (i, name) in atom_name.iter().enumerate() {
                for (j, j_name) in proj_name.iter().enumerate() {
                    let name = AtomType::from_str(name);
                    if name.as_ref().ok() == j_name.as_ref().ok() && name.is_ok() && j_name.is_ok() {
                        let use_atom = Atom::new(atom_pos.row(i).to_owned(), proj_list[j], name.unwrap().clone());
                        orb_proj.extend(atom_proj[j].clone());
                        atom.push(use_atom.clone());
                        for _ in 0..proj_list[j] {
                            orb.push_row(use_atom.position().view());
                        }
                    }
                }
            }
            orb
        };
        //开始尝试读取 _r.dat 文件
        let mut reads: Vec<String> = Vec::new();
        let mut r_path = file_path.clone();
        r_path.push_str("_r.dat");
        let path = Path::new(&r_path);
        let hr = File::open(path);
        let mut have_r = false;
        let mut rmatrix = if let Ok(hr) = hr {
            have_r = true;
            let reader = BufReader::new(hr);
            let mut reads: Vec<String> = Vec::new();
            for line in reader.lines() {
                let line = line.map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to read line: {}", e) })?;
                reads.push(line.clone());
            }
            let n_R = reads[2].trim().parse::<usize>()
                .map_err(|e| TbError::FileParse { file: r_path.clone(), message: format!("Failed to parse n_R: {}", e) })?;
            let mut rmatrix = Array4::<Complex<f64>>::zeros((hamR.nrows(), 3, nsta, nsta));
            for i in 0..n_R {
                let mut string = reads[i * nsta * nsta + 3].trim().split_whitespace();
                let a = string.next().ok_or_else(|| TbError::FileParse { file: r_path.clone(), message: "Missing R vector component".to_string() })?
                    .parse::<isize>().map_err(|e| TbError::FileParse { file: r_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                let b = string.next().ok_or_else(|| TbError::FileParse { file: r_path.clone(), message: "Missing R vector component".to_string() })?
                    .parse::<isize>().map_err(|e| TbError::FileParse { file: r_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                let c = string.next().ok_or_else(|| TbError::FileParse { file: r_path.clone(), message: "Missing R vector component".to_string() })?
                    .parse::<isize>().map_err(|e| TbError::FileParse { file: r_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                let R0 = array![a, b, c];
                let index = find_R(&hamR, &R0)
                    .ok_or_else(|| TbError::FileParse { file: r_path.clone(), message: format!("R vector {:?} not found in Hamiltonian", R0) })?;
                for ind_i in 0..nsta {
                    for ind_j in 0..nsta {
                        let string = &reads[i * nsta * nsta + ind_i * nsta + ind_j + 3];
                        let mut string = string.trim().split_whitespace();
                        string.nth(4);
                        for r in 0..3 {
                            let re = string.next().ok_or_else(|| TbError::FileParse { file: r_path.clone(), message: "Missing R matrix real part".to_string() })?
                                .parse::<f64>().map_err(|e| TbError::FileParse { file: r_path.clone(), message: format!("Failed to parse R matrix real part: {}", e) })?;
                            let im = string.next().ok_or_else(|| TbError::FileParse { file: r_path.clone(), message: "Missing R matrix imaginary part".to_string() })?
                                .parse::<f64>().map_err(|e| TbError::FileParse { file: r_path.clone(), message: format!("Failed to parse R matrix imaginary part: {}", e) })?;
                            rmatrix[[index, r, ind_j, ind_i]] =
                                Complex::new(re, im) / (weights[i] as f64);
                        }
                    }
                }
            }
            rmatrix
        } else {
            let mut rmatrix = Array4::<Complex<f64>>::zeros((1, 3, nsta, nsta));
            for i in 0..norb {
                for r in 0..3 {
                    rmatrix[[0, r, i, i]] = Complex::<f64>::from(orb[[i, r]]);
                    if spin {
                        rmatrix[[0, r, i + norb, i + norb]] = Complex::<f64>::from(orb[[i, r]]);
                    }
                }
            }
            rmatrix
        };

        //最后判断有没有wannier90_wsvec.dat-----------------------------------
        let mut ws_path = file_path.clone();
        ws_path.push_str("_wsvec.dat");
        let path = Path::new(&ws_path); //转化为路径格式
        let ws = File::open(path);
        let mut have_ws = false;
        if let Ok(ws) = ws {
            have_ws = true;
            let reader = BufReader::new(ws);
            let mut reads: Vec<String> = Vec::new();
            for line in reader.lines() {
                let line = line.map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to read line: {}", e) })?;
                reads.push(line.clone());
            }
            //开始针对ham, hamR 以及 rmatrix 进行修改
            //我们先考虑有rmatrix的情况
            if have_r {
                let mut i = 0;
                let mut new_hamR = Array2::zeros((1, 3));
                let mut new_ham = Array3::zeros((1, nsta, nsta));
                let mut new_rmatrix = Array4::zeros((1, 3, nsta, nsta));
                while (i < reads.len() - 1) {
                    i += 1;
                    let line = &reads[i];
                    let mut string = line.trim().split_whitespace();
                    let a = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing R vector component".to_string() })?
                        .parse::<isize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                    let b = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing R vector component".to_string() })?
                        .parse::<isize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                    let c = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing R vector component".to_string() })?
                        .parse::<isize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                    let int_i = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing orbital index".to_string() })?
                        .parse::<usize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse orbital index: {}", e) })? - 1;
                    let int_j = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing orbital index".to_string() })?
                        .parse::<usize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse orbital index: {}", e) })? - 1;
                    //接下来判断是否在我们的hamR 中
                    i += 1;
                    let mut weight = reads[i]
                        .trim()
                        .split_whitespace()
                        .next()
                        .ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing weight value".to_string() })?
                        .parse::<usize>()
                        .map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse weight: {}", e) })?;
                    let R = array![a, b, c];
                    let index = find_R(&hamR, &R)
                        .ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: format!("R vector {:?} not found in Hamiltonian", R) })?;
                    let hop = ham[[index, int_i, int_j]] / (weight as f64);
                    let hop_x = rmatrix[[index, 0, int_i, int_j]] / (weight as f64);
                    let hop_y = rmatrix[[index, 1, int_i, int_j]] / (weight as f64);
                    let hop_z = rmatrix[[index, 2, int_i, int_j]] / (weight as f64);

                    for i0 in 0..weight {
                        i += 1;
                        let line = &reads[i];
                        let mut string = line.trim().split_whitespace();
                        let a = string.next().unwrap().parse::<isize>().unwrap();
                        let b = string.next().unwrap().parse::<isize>().unwrap();
                        let c = string.next().unwrap().parse::<isize>().unwrap();
                        let new_R = array![R[[0]] + a, R[[1]] + b, R[[2]] + c];
                        if let Some(index0)= find_R(&new_hamR, &new_R) {
                            new_ham[[index0, int_i, int_j]] += hop;
                            new_rmatrix[[index0, 0, int_i, int_j]] += hop_x;
                            new_rmatrix[[index0, 1, int_i, int_j]] += hop_y;
                            new_rmatrix[[index0, 2, int_i, int_j]] += hop_z;
                        } else {
                            let mut use_ham = Array2::zeros((nsta, nsta));
                            let mut use_rmatrix = Array3::zeros((3, nsta, nsta));
                            use_ham[[int_i, int_j]] += hop;
                            use_rmatrix[[0, int_i, int_j]] += hop_x;
                            use_rmatrix[[1, int_i, int_j]] += hop_y;
                            use_rmatrix[[2, int_i, int_j]] += hop_z;
                            new_hamR.push_row(new_R.view());
                            new_ham.push(Axis(0), use_ham.view());
                            new_rmatrix.push(Axis(0), use_rmatrix.view());
                        }
                    }
                }
                hamR = new_hamR;
                ham = new_ham;
                rmatrix = new_rmatrix;
            } else {
                let mut i = 0;
                let mut new_hamR = Array2::zeros((1, 3));
                let mut new_ham = Array3::zeros((1, nsta, nsta));
                while (i < reads.len() - 1) {
                    i += 1;
                    let line = &reads[i];
                    let mut string = line.trim().split_whitespace();
                    let a = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing R vector component".to_string() })?
                        .parse::<isize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                    let b = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing R vector component".to_string() })?
                        .parse::<isize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                    let c = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing R vector component".to_string() })?
                        .parse::<isize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                    let int_i = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing orbital index".to_string() })?
                        .parse::<usize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse orbital index: {}", e) })? - 1;
                    let int_j = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing orbital index".to_string() })?
                        .parse::<usize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse orbital index: {}", e) })? - 1;
                    //接下来判断是否在我们的hamR 中
                    i += 1;
                    let mut weight = reads[i]
                        .trim()
                        .split_whitespace()
                        .next()
                        .ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing weight value".to_string() })?
                        .parse::<usize>()
                        .map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse weight: {}", e) })?;
                    let R = array![a, b, c];
                    let index = find_R(&hamR, &R)
                        .ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: format!("R vector {:?} not found in Hamiltonian", R) })?;
                    let hop = ham[[index, int_i, int_j]] / (weight as f64);

                    for i0 in 0..weight {
                        i += 1;
                        let line = &reads[i];
                        let mut string = line.trim().split_whitespace();
                        let a = string.next().unwrap().parse::<isize>().unwrap();
                        let b = string.next().unwrap().parse::<isize>().unwrap();
                        let c = string.next().unwrap().parse::<isize>().unwrap();
                        let new_R = array![R[[0]] + a, R[[1]] + b, R[[2]] + c];
                        if let Some(index0)=find_R(&new_hamR, &new_R) {
                            new_ham[[index0, int_i, int_j]] += hop;
                        } else {
                            let mut use_ham = Array2::zeros((nsta, nsta));
                            use_ham[[int_i, int_j]] = hop;
                            new_hamR.push_row(new_R.view());
                            new_ham.push(Axis(0), use_ham.view());
                        }
                    }
                }
                hamR = new_hamR;
                ham = new_ham;
            }
        }
        //最后一步, 将rmatrix 变成厄密的

        if have_r {
            for r in 0..hamR.nrows() - 1 {
                let R = hamR.row(r);
                let R_inv = -&R;
                if let Some(index)=find_R(&hamR, &R_inv) {
                    for i in 0..nsta {
                        for j in 0..nsta {
                            rmatrix[[r, 0, i, j]] =
                                (rmatrix[[r, 0, i, j]] + rmatrix[[index, 0, j, i]].conj()) / 2.0;
                            rmatrix[[r, 1, i, j]] =
                                (rmatrix[[r, 1, i, j]] + rmatrix[[index, 1, j, i]].conj()) / 2.0;
                            rmatrix[[r, 2, i, j]] =
                                (rmatrix[[r, 2, i, j]] + rmatrix[[index, 2, j, i]].conj()) / 2.0;
                            rmatrix[[index, 0, j, i]] = rmatrix[[r, 0, i, j]].conj();
                            rmatrix[[index, 1, j, i]] = rmatrix[[r, 1, i, j]].conj();
                            rmatrix[[index, 2, j, i]] = rmatrix[[r, 2, i, j]].conj();
                        }
                    }
                } else {
                    return Err(TbError::MissingHermitianConjugate { r: R.to_owned() });
                }
            }
        }

        let mut model = Model {
            dim_r: 3,
            spin,
            lat,
            orb,
            orb_projection: orb_proj,
            atoms: atom,
            ham,
            hamR,
            rmatrix,
        };
        Ok(model)
    }

    #[allow(non_snake_case)]
    pub fn from_tb(path: &str, file_name: &str, zero_energy: f64) -> Result<Model> {
        //! 这个函数是从 wannier90 中读取 TB 文件的
        //!
        //! 这里 path 表示 文件的位置, 可以使用绝对路径, 即 "/" 开头的路径, 也可以使用相对路径, 即运行 cargo run 时
        //! 候的文件夹作为起始路径.
        //!
        //! file_name 就是 wannier90 中的 seedname, 文件可以读取
        //! seedname.win, seedname_centres.xyz, seedname_hr.dat, 以及可选的 seedname_r.dat 以及 seedname_wsved.dat
        //!
        //! 这里 seedname_centres.xyz 需要在 wannier90 中设置 write_xyz=true, 而
        //! seedname_tb.dat 需要设置 write_tb=true.
        //! 如果想要计算输运性质, 则需要 write_rmn=true, 能够给出一个seedname_r.dat.
        //!
        //! 此外, 对于高版本的wannier90, 如果想要保持较好的对称性, 建议将wannier90_wsvec.dat
        //! 也给出, 这样能够得到较好的对称结果

        use std::fs::File;
        use std::io::BufRead;
        use std::io::BufReader;
        use std::path::Path;

        let mut file_path = path.to_string();
        file_path.push_str(file_name);
        let mut hr_path = file_path.clone();
        hr_path.push_str("_hr.dat");

        let path = Path::new(&hr_path);
        let hr = File::open(path).map_err(|e| TbError::FileCreation {
            path: hr_path.clone(),
            message: format!("Unable to open HR file: {}", e),
        })?;
        let reader = BufReader::new(hr);
        let mut reads: Vec<String> = Vec::new();

        // 读取文件行
        for line in reader.lines() {
            let line = line.map_err(|e| TbError::Io(e))?;
            reads.push(line.clone());
        }

        // 获取轨道数和R点数
        let nsta = reads[1].trim().parse::<usize>()
            .map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse nsta: {}", e) })?;
        let n_R = reads[2].trim().parse::<usize>()
            .map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse n_R: {}", e) })?;
        let mut weights: Vec<usize> = Vec::new();
        let mut n_line: usize = 0;

        // 解析文件数据以获取权重
        for i in 3..reads.len() {
            if reads[i].contains(".") {
                n_line = i;
                break;
            }
            let string = reads[i].trim().split_whitespace();
            let string: Vec<_> = string.map(|x| x.parse::<usize>()
                .map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse weight: {}", e) }))
                .collect::<Result<Vec<_>>>()?;
            weights.extend(string.clone());
        }

        // 初始化哈密顿量矩阵
        let mut hamR = Array2::<isize>::zeros((1, 3));
        let mut ham = Array3::<Complex<f64>>::zeros((1, nsta, nsta));

        // 遍历每个R点并填充哈密顿量
        for i in 0..n_R {
            let mut string = reads[i * nsta * nsta + n_line].trim().split_whitespace();
            let a = string.next().ok_or_else(|| TbError::FileParse { file: hr_path.clone(), message: "Missing R vector component".to_string() })?
                .parse::<isize>().map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
            let b = string.next().ok_or_else(|| TbError::FileParse { file: hr_path.clone(), message: "Missing R vector component".to_string() })?
                .parse::<isize>().map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
            let c = string.next().ok_or_else(|| TbError::FileParse { file: hr_path.clone(), message: "Missing R vector component".to_string() })?
                .parse::<isize>().map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;

            if a == 0 && b == 0 && c == 0 {
                for ind_i in 0..nsta {
                    for ind_j in 0..nsta {
                        let mut string = reads[i * nsta * nsta + ind_i * nsta + ind_j + n_line]
                            .trim()
                            .split_whitespace();
                        let re = string.nth(5).ok_or_else(|| TbError::FileParse { file: hr_path.clone(), message: "Missing Hamiltonian real part".to_string() })?
                            .parse::<f64>().map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse Hamiltonian real part: {}", e) })?;
                        let im = string.next().ok_or_else(|| TbError::FileParse { file: hr_path.clone(), message: "Missing Hamiltonian imaginary part".to_string() })?
                            .parse::<f64>().map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse Hamiltonian imaginary part: {}", e) })?;
                        ham[[0, ind_j, ind_i]] = Complex::new(re, im) / (weights[i] as f64);
                    }
                }
            } else {
                let mut matrix = Array3::<Complex<f64>>::zeros((1, nsta, nsta));
                for ind_i in 0..nsta {
                    for ind_j in 0..nsta {
                        let mut string = reads[i * nsta * nsta + ind_i * nsta + ind_j + n_line]
                            .trim()
                            .split_whitespace();
                        let re = string.nth(5).ok_or_else(|| TbError::FileParse { file: hr_path.clone(), message: "Missing Hamiltonian real part".to_string() })?
                            .parse::<f64>().map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse Hamiltonian real part: {}", e) })?;
                        let im = string.next().ok_or_else(|| TbError::FileParse { file: hr_path.clone(), message: "Missing Hamiltonian imaginary part".to_string() })?
                            .parse::<f64>().map_err(|e| TbError::FileParse { file: hr_path.clone(), message: format!("Failed to parse Hamiltonian imaginary part: {}", e) })?;
                        matrix[[0, ind_j, ind_i]] = Complex::new(re, im) / (weights[i] as f64);
                        // wannier90 里面是按照纵向排列的矩阵
                    }
                }
                ham.append(Axis(0), matrix.view())
                    .map_err(|e| TbError::Linalg(ndarray_linalg::error::LinalgError::Shape(e)))?;
                hamR.append(Axis(0), arr2(&[[a, b, c]]).view())
                    .map_err(|e| TbError::Linalg(ndarray_linalg::error::LinalgError::Shape(e)))?;
            }
        }

        // 调整哈密顿量以匹配能量零点
        for i in 0..nsta {
            ham[[0, i, i]] -= Complex::new(zero_energy, 0.0);
        }
        //开始读取 .win 文件
        let mut reads: Vec<String> = Vec::new();
        let mut win_path = file_path.clone();
        win_path.push_str(".win"); //文件的位置
        let path = Path::new(&win_path); //转化为路径格式
        let hr = File::open(path).map_err(|e| TbError::FileCreation {
            path: win_path.clone(),
            message: format!("Unable to open win file: {}", e),
        })?;
        let reader = BufReader::new(hr);
        let mut reads: Vec<String> = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| TbError::Io(e))?;
            reads.push(line.clone());
        }
        let mut read_iter = reads.iter();
        let mut lat = Array2::<f64>::zeros((3, 3)); //晶格轨道坐标初始化
        let mut spin: bool = false; //体系自旋初始化
        let mut natom: usize = 0; //原子位置初始化
        let mut atom = Vec::new(); //原子位置坐标初始化
        let mut orb_proj = Vec::new();
        let mut proj_name = Vec::new();
        let mut proj_list: Vec<usize> = Vec::new();
        let mut atom_list: Vec<usize> = Vec::new();
        let mut atom_name: Vec<&str> = Vec::new();
        let mut atom_pos = Array2::<f64>::zeros((0, 3));
        let mut atom_proj = Vec::new();
        let mut norb = 0;
        loop {
            let a = read_iter.next();
            if a == None {
                break;
            } else {
                let a = a.ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Unexpected end of file".to_string() })?;
                if a.contains("begin unit_cell_cart") {
                    let mut lat1 = read_iter.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Missing lattice vector line".to_string() })?.trim().split_whitespace(); //将数字放到
                    let mut lat2 = read_iter.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Missing lattice vector line".to_string() })?.trim().split_whitespace();
                    let mut lat3 = read_iter.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Missing lattice vector line".to_string() })?.trim().split_whitespace();
                    for i in 0..3 {
                        lat[[0, i]] = lat1.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Missing lattice vector component".to_string() })?
                            .parse::<f64>().map_err(|e| TbError::FileParse { file: win_path.clone(), message: format!("Failed to parse lattice vector: {}", e) })?;
                        lat[[1, i]] = lat2.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Missing lattice vector component".to_string() })?
                            .parse::<f64>().map_err(|e| TbError::FileParse { file: win_path.clone(), message: format!("Failed to parse lattice vector: {}", e) })?;
                        lat[[2, i]] = lat3.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Missing lattice vector component".to_string() })?
                            .parse::<f64>().map_err(|e| TbError::FileParse { file: win_path.clone(), message: format!("Failed to parse lattice vector: {}", e) })?;
                    }
                } else if a.contains("spinors") && (a.contains("T") || a.contains("t")) {
                    spin = true;
                } else if a.contains("begin projections") {
                    loop {
                        let string = read_iter.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Unexpected end of file".to_string() })?;
                        if string.contains("end projections") {
                            break;
                        } else {
                            let prj: Vec<&str> = string
                                .split(|c| c == ',' || c == ';' || c == ':')
                                .map(|x| x.trim())
                                .collect();
                            let mut atom_orb_number: usize = 0;
                            let mut proj_orb = Vec::new();
                            for item in prj[1..].iter() {
                                let (aa, use_proj_orb): (usize, Vec<_>) = match (*item).trim() {
                                    "s" => (1, vec![OrbProj::s]),
                                    "p" => (3, vec![OrbProj::pz, OrbProj::px, OrbProj::py]),
                                    "d" => (
                                        5,
                                        vec![
                                            OrbProj::dz2,
                                            OrbProj::dxz,
                                            OrbProj::dyz,
                                            OrbProj::dx2y2,
                                            OrbProj::dxy,
                                        ],
                                    ),
                                    "f" => (
                                        7,
                                        vec![
                                            OrbProj::fz3,
                                            OrbProj::fxz2,
                                            OrbProj::fyz2,
                                            OrbProj::fzx2y2,
                                            OrbProj::fxyz,
                                            OrbProj::fxx23y2,
                                            OrbProj::fy3x2y2,
                                        ],
                                    ),
                                    "sp3" => (
                                        4,
                                        vec![
                                            OrbProj::sp3_1,
                                            OrbProj::sp3_2,
                                            OrbProj::sp3_3,
                                            OrbProj::sp3_4,
                                        ],
                                    ),
                                    "sp2" => {
                                        (3, vec![OrbProj::sp2_1, OrbProj::sp2_2, OrbProj::sp2_3])
                                    }
                                    "sp" => (2, vec![OrbProj::sp_1, OrbProj::sp_2]),
                                    "sp3d" => (
                                        5,
                                        vec![
                                            OrbProj::sp3d_1,
                                            OrbProj::sp3d_2,
                                            OrbProj::sp3d_3,
                                            OrbProj::sp3d_4,
                                            OrbProj::sp3d_5,
                                        ],
                                    ),
                                    "sp3d2" => (
                                        6,
                                        vec![
                                            OrbProj::sp3d2_1,
                                            OrbProj::sp3d2_2,
                                            OrbProj::sp3d2_3,
                                            OrbProj::sp3d2_4,
                                            OrbProj::sp3d2_5,
                                            OrbProj::sp3d2_6,
                                        ],
                                    ),
                                    "px" => (1, vec![OrbProj::px]),
                                    "py" => (1, vec![OrbProj::py]),
                                    "pz" => (1, vec![OrbProj::pz]),
                                    "dxy" => (1, vec![OrbProj::dxy]),
                                    "dxz" => (1, vec![OrbProj::dxz]),
                                    "dyz" => (1, vec![OrbProj::dyz]),
                                    "dz2" => (1, vec![OrbProj::dz2]),
                                    "dx2-y2" => (1, vec![OrbProj::dx2y2]),
                                    &_ => return Err(TbError::InvalidOrbitalProjection(
                                        format!("Unrecognized projection '{}' in seedname.win", item)
                                    )),
                                };
                                atom_orb_number += aa;
                                proj_orb.extend(use_proj_orb);
                            }
                            proj_list.push(atom_orb_number);
                            atom_proj.push(proj_orb);
                            let proj_type = AtomType::from_str(prj[0]);
                            proj_name.push(proj_type);
                        }
                    }
                } else if a.contains("begin atoms_cart") {
                    loop {
                        let string = read_iter.next().ok_or_else(|| TbError::FileParse { file: win_path.clone(), message: "Unexpected end of file".to_string() })?;
                        if string.contains("end atoms_cart") {
                            break;
                        } else {
                            let prj: Vec<&str> = string.split_whitespace().collect();
                            atom_name.push(prj[0]);
                            let a1 = prj[1].parse::<f64>()
                                .map_err(|e| TbError::FileParse { file: win_path.clone(), message: format!("Failed to parse atom position: {}", e) })?;
                            let a2 = prj[2].parse::<f64>()
                                .map_err(|e| TbError::FileParse { file: win_path.clone(), message: format!("Failed to parse atom position: {}", e) })?;
                            let a3 = prj[3].parse::<f64>()
                                .map_err(|e| TbError::FileParse { file: win_path.clone(), message: format!("Failed to parse atom position: {}", e) })?;
                            let a = array![a1, a2, a3];
                            atom_pos.push_row(a.view()); //这里我们不用win 里面的, 因为这个和orb没法对应, 如果没有xyz文件才考虑用这个
                        }
                    }
                }
            }
        }
        //开始读取 seedname_centres.xyz 文件
        let mut reads: Vec<String> = Vec::new();
        let mut xyz_path = file_path.clone();
        xyz_path.push_str("_centres.xyz");
        let path = Path::new(&xyz_path);
        let hr = File::open(path);
        let orb = if let Ok(hr) = hr {
            let reader = BufReader::new(hr);
            let mut reads: Vec<String> = Vec::new();
            for line in reader.lines() {
                let line = line.map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to read line: {}", e) })?;
                reads.push(line.clone());
            }
            //let nsta=reads[0].trim().parse::<usize>().unwrap()-natom;
            let norb = if spin { nsta / 2 } else { nsta };
            let mut orb = Array2::<f64>::zeros((norb, 3));
            for i in 0..norb {
                let a: Vec<&str> = reads[i + 2].trim().split_whitespace().collect();
                orb[[i, 0]] = a[1].parse::<f64>()
                    .map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to parse orbital position: {}", e) })?;
                orb[[i, 1]] = a[2].parse::<f64>()
                    .map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to parse orbital position: {}", e) })?;
                orb[[i, 2]] = a[3].parse::<f64>()
                    .map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to parse orbital position: {}", e) })?
            }
            orb = orb.dot(&lat.inv().map_err(TbError::Linalg)?);
            let mut new_atom_pos = Array2::<f64>::zeros((reads.len() - 2 - nsta, 3));
            let mut new_atom_name = Vec::new();
            for i in 0..reads.len() - 2 - nsta {
                let a: Vec<&str> = reads[i + 2 + nsta].trim().split_whitespace().collect();
                new_atom_pos[[i, 0]] = a[1].parse::<f64>()
                    .map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to parse atom position: {}", e) })?;
                new_atom_pos[[i, 1]] = a[2].parse::<f64>()
                    .map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to parse atom position: {}", e) })?;
                new_atom_pos[[i, 2]] = a[3].parse::<f64>()
                    .map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to parse atom position: {}", e) })?;
                new_atom_name.push(AtomType::from_str(a[0]));
            }
            //接下来如果wannier90.win 和 .xyz 文件的原子顺序不一致, 那么我们以xyz的原子顺序为准, 调整 atom_list

            for (i, name) in new_atom_name.iter().enumerate() {
                for (j, j_name) in proj_name.iter().enumerate() {
                    if name.as_ref().ok() == j_name.as_ref().ok() && name.is_ok() && j_name.is_ok() {
                        let use_pos = new_atom_pos.row(i).dot(&lat.inv().map_err(TbError::Linalg)?);
                        let use_atom = Atom::new(use_pos, proj_list[j], name.as_ref().unwrap().clone());
                        atom.push(use_atom);
                        orb_proj.extend(atom_proj[j].clone());
                    }
                }
            }
            orb
        } else {
            let mut orb = Array2::<f64>::zeros((0, 3));
            let atom_pos = atom_pos.dot(&lat.inv().map_err(TbError::Linalg)?);
            for (i, name) in atom_name.iter().enumerate() {
                for (j, j_name) in proj_name.iter().enumerate() {
                    let name = AtomType::from_str(name);
                    if name.as_ref().ok() == j_name.as_ref().ok() && name.is_ok() && j_name.is_ok() {
                        let use_atom = Atom::new(atom_pos.row(i).to_owned(), proj_list[j], name.unwrap().clone());
                        orb_proj.extend(atom_proj[j].clone());
                        atom.push(use_atom.clone());
                        for _ in 0..proj_list[j] {
                            orb.push_row(use_atom.position().view());
                        }
                    }
                }
            }
            orb
        };
        //开始尝试读取 _r.dat 文件
        let mut reads: Vec<String> = Vec::new();
        let mut r_path = file_path.clone();
        r_path.push_str("_r.dat");
        let path = Path::new(&r_path);
        let hr = File::open(path);
        let mut have_r = false;
        let mut rmatrix = if let Ok(hr) = hr {
            have_r = true;
            let reader = BufReader::new(hr);
            let mut reads: Vec<String> = Vec::new();
            for line in reader.lines() {
                let line = line.map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to read line: {}", e) })?;
                reads.push(line.clone());
            }
            let n_R = reads[2].trim().parse::<usize>()
                .map_err(|e| TbError::FileParse { file: r_path.clone(), message: format!("Failed to parse n_R: {}", e) })?;
            let mut rmatrix = Array4::<Complex<f64>>::zeros((hamR.nrows(), 3, nsta, nsta));
            for i in 0..n_R {
                let mut string = reads[i * nsta * nsta + 3].trim().split_whitespace();
                let a = string.next().ok_or_else(|| TbError::FileParse { file: r_path.clone(), message: "Missing R vector component".to_string() })?
                    .parse::<isize>().map_err(|e| TbError::FileParse { file: r_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                let b = string.next().ok_or_else(|| TbError::FileParse { file: r_path.clone(), message: "Missing R vector component".to_string() })?
                    .parse::<isize>().map_err(|e| TbError::FileParse { file: r_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                let c = string.next().ok_or_else(|| TbError::FileParse { file: r_path.clone(), message: "Missing R vector component".to_string() })?
                    .parse::<isize>().map_err(|e| TbError::FileParse { file: r_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                let R0 = array![a, b, c];
                let index = find_R(&hamR, &R0)
                    .ok_or_else(|| TbError::FileParse { file: r_path.clone(), message: format!("R vector {:?} not found in Hamiltonian", R0) })?;
                for ind_i in 0..nsta {
                    for ind_j in 0..nsta {
                        let string = &reads[i * nsta * nsta + ind_i * nsta + ind_j + 3];
                        let mut string = string.trim().split_whitespace();
                        string.nth(4);
                        for r in 0..3 {
                            let re = string.next().ok_or_else(|| TbError::FileParse { file: r_path.clone(), message: "Missing R matrix real part".to_string() })?
                                .parse::<f64>().map_err(|e| TbError::FileParse { file: r_path.clone(), message: format!("Failed to parse R matrix real part: {}", e) })?;
                            let im = string.next().ok_or_else(|| TbError::FileParse { file: r_path.clone(), message: "Missing R matrix imaginary part".to_string() })?
                                .parse::<f64>().map_err(|e| TbError::FileParse { file: r_path.clone(), message: format!("Failed to parse R matrix imaginary part: {}", e) })?;
                            rmatrix[[index, r, ind_j, ind_i]] =
                                Complex::new(re, im) / (weights[i] as f64);
                        }
                    }
                }
            }
            rmatrix
        } else {
            let mut rmatrix = Array4::<Complex<f64>>::zeros((1, 3, nsta, nsta));
            for i in 0..norb {
                for r in 0..3 {
                    rmatrix[[0, r, i, i]] = Complex::<f64>::from(orb[[i, r]]);
                    if spin {
                        rmatrix[[0, r, i + norb, i + norb]] = Complex::<f64>::from(orb[[i, r]]);
                    }
                }
            }
            rmatrix
        };

        //最后判断有没有wannier90_wsvec.dat-----------------------------------
        let mut ws_path = file_path.clone();
        ws_path.push_str("_wsvec.dat");
        let path = Path::new(&ws_path); //转化为路径格式
        let ws = File::open(path);
        let mut have_ws = false;
        if let Ok(ws) = ws {
            have_ws = true;
            let reader = BufReader::new(ws);
            let mut reads: Vec<String> = Vec::new();
            for line in reader.lines() {
                let line = line.map_err(|e| TbError::FileParse { file: xyz_path.clone(), message: format!("Failed to read line: {}", e) })?;
                reads.push(line.clone());
            }
            //开始针对ham, hamR 以及 rmatrix 进行修改
            //我们先考虑有rmatrix的情况
            if have_r {
                let mut i = 0;
                let mut new_hamR = Array2::zeros((1, 3));
                let mut new_ham = Array3::zeros((1, nsta, nsta));
                let mut new_rmatrix = Array4::zeros((1, 3, nsta, nsta));
                while (i < reads.len() - 1) {
                    i += 1;
                    let line = &reads[i];
                    let mut string = line.trim().split_whitespace();
                    let a = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing R vector component".to_string() })?
                        .parse::<isize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                    let b = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing R vector component".to_string() })?
                        .parse::<isize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                    let c = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing R vector component".to_string() })?
                        .parse::<isize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                    let int_i = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing orbital index".to_string() })?
                        .parse::<usize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse orbital index: {}", e) })? - 1;
                    let int_j = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing orbital index".to_string() })?
                        .parse::<usize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse orbital index: {}", e) })? - 1;
                    //接下来判断是否在我们的hamR 中
                    i += 1;
                    let mut weight = reads[i]
                        .trim()
                        .split_whitespace()
                        .next()
                        .ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing weight value".to_string() })?
                        .parse::<usize>()
                        .map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse weight: {}", e) })?;
                    let R = array![a, b, c];
                    let index = find_R(&hamR, &R)
                        .ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: format!("R vector {:?} not found in Hamiltonian", R) })?;
                    let hop = ham[[index, int_i, int_j]] / (weight as f64);
                    let hop_x = rmatrix[[index, 0, int_i, int_j]] / (weight as f64);
                    let hop_y = rmatrix[[index, 1, int_i, int_j]] / (weight as f64);
                    let hop_z = rmatrix[[index, 2, int_i, int_j]] / (weight as f64);

                    for i0 in 0..weight {
                        i += 1;
                        let line = &reads[i];
                        let mut string = line.trim().split_whitespace();
                        let a = string.next().unwrap().parse::<isize>().unwrap();
                        let b = string.next().unwrap().parse::<isize>().unwrap();
                        let c = string.next().unwrap().parse::<isize>().unwrap();
                        let new_R = array![R[[0]] + a, R[[1]] + b, R[[2]] + c];
                        if let Some(index0)=find_R(&new_hamR, &new_R) {
                            new_ham[[index0, int_i, int_j]] += hop;
                            new_rmatrix[[index0, 0, int_i, int_j]] += hop_x;
                            new_rmatrix[[index0, 1, int_i, int_j]] += hop_y;
                            new_rmatrix[[index0, 2, int_i, int_j]] += hop_z;
                        } else {
                            let mut use_ham = Array2::zeros((nsta, nsta));
                            let mut use_rmatrix = Array3::zeros((3, nsta, nsta));
                            use_ham[[int_i, int_j]] += hop;
                            use_rmatrix[[0, int_i, int_j]] += hop_x;
                            use_rmatrix[[1, int_i, int_j]] += hop_y;
                            use_rmatrix[[2, int_i, int_j]] += hop_z;
                            new_hamR.push_row(new_R.view());
                            new_ham.push(Axis(0), use_ham.view());
                            new_rmatrix.push(Axis(0), use_rmatrix.view());
                        }
                    }
                }
                hamR = new_hamR;
                ham = new_ham;
                rmatrix = new_rmatrix;
            } else {
                let mut i = 0;
                let mut new_hamR = Array2::zeros((1, 3));
                let mut new_ham = Array3::zeros((1, nsta, nsta));
                while (i < reads.len() - 1) {
                    i += 1;
                    let line = &reads[i];
                    let mut string = line.trim().split_whitespace();
                    let a = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing R vector component".to_string() })?
                        .parse::<isize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                    let b = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing R vector component".to_string() })?
                        .parse::<isize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                    let c = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing R vector component".to_string() })?
                        .parse::<isize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse R vector: {}", e) })?;
                    let int_i = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing orbital index".to_string() })?
                        .parse::<usize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse orbital index: {}", e) })? - 1;
                    let int_j = string.next().ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing orbital index".to_string() })?
                        .parse::<usize>().map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse orbital index: {}", e) })? - 1;
                    //接下来判断是否在我们的hamR 中
                    i += 1;
                    let mut weight = reads[i]
                        .trim()
                        .split_whitespace()
                        .next()
                        .ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: "Missing weight value".to_string() })?
                        .parse::<usize>()
                        .map_err(|e| TbError::FileParse { file: ws_path.clone(), message: format!("Failed to parse weight: {}", e) })?;
                    let R = array![a, b, c];
                    let index = find_R(&hamR, &R)
                        .ok_or_else(|| TbError::FileParse { file: ws_path.clone(), message: format!("R vector {:?} not found in Hamiltonian", R) })?;
                    let hop = ham[[index, int_i, int_j]] / (weight as f64);

                    for i0 in 0..weight {
                        i += 1;
                        let line = &reads[i];
                        let mut string = line.trim().split_whitespace();
                        let a = string.next().unwrap().parse::<isize>().unwrap();
                        let b = string.next().unwrap().parse::<isize>().unwrap();
                        let c = string.next().unwrap().parse::<isize>().unwrap();
                        let new_R = array![R[[0]] + a, R[[1]] + b, R[[2]] + c];
                        if let Some(index0)=find_R(&new_hamR, &new_R) {
                            new_ham[[index0, int_i, int_j]] += hop;
                        } else {
                            let mut use_ham = Array2::zeros((nsta, nsta));
                            use_ham[[int_i, int_j]] = hop;
                            new_hamR.push_row(new_R.view());
                            new_ham.push(Axis(0), use_ham.view());
                        }
                    }
                }
                hamR = new_hamR;
                ham = new_ham;
            }
        }
        //最后一步, 将rmatrix 变成厄密的

        if have_r {
            for r in 0..hamR.nrows() - 1 {
                let R = hamR.row(r);
                let R_inv = -&R;
                if let Some(index)=find_R(&hamR, &R_inv) {
                    for i in 0..nsta {
                        for j in 0..nsta {
                            rmatrix[[r, 0, i, j]] =
                                (rmatrix[[r, 0, i, j]] + rmatrix[[index, 0, j, i]].conj()) / 2.0;
                            rmatrix[[r, 1, i, j]] =
                                (rmatrix[[r, 1, i, j]] + rmatrix[[index, 1, j, i]].conj()) / 2.0;
                            rmatrix[[r, 2, i, j]] =
                                (rmatrix[[r, 2, i, j]] + rmatrix[[index, 2, j, i]].conj()) / 2.0;
                            rmatrix[[index, 0, j, i]] = rmatrix[[r, 0, i, j]].conj();
                            rmatrix[[index, 1, j, i]] = rmatrix[[r, 1, i, j]].conj();
                            rmatrix[[index, 2, j, i]] = rmatrix[[r, 2, i, j]].conj();
                        }
                    }
                } else {
                    return Err(TbError::MissingHermitianConjugate { r: R.to_owned() });
                }
            }
        }

        let mut model = Model {
            dim_r: 3,
            spin,
            lat,
            orb,
            orb_projection: orb_proj,
            atoms: atom,
            ham,
            hamR,
            rmatrix,
        };
        Ok(model)
    }
}
