extern crate neuroflow;
extern crate rand;

use neuroflow::FeedForward;
use neuroflow::data::DataSet;
use neuroflow::activators::Type::Tanh;
//use neuroflow::activators::Type::Sigmoid;
use std::fs::File;
use std::io::Read;
use std::env;
use rand::prelude::*;


fn usage() {
    println!("USAGE:");
    println!("\t[mode] [file1] [file2]");
    println!("EXAMPLES:");
    println!("\tmake");
    println!("\ttarget/release/neural train intersting_strings.txt model.ai");
    println!("\ttarget/release/neural exec model.ai file_to_parse.exe");
    std::process::exit(1);
}

fn load(filename: &str) -> Vec<u8> {
    let mut fd = File::open(filename).expect("provide a file");
    let mut data: Vec<u8> = Vec::new();
    fd.read_to_end(&mut data).expect("cannot read the file");

    data
}

fn split_bytes_float(raw: &Vec<u8>) -> Vec<Vec<f64>> {
    let mut spl: Vec<Vec<f64>> = Vec::new();
    let mut buff: Vec<f64> = Vec::new();

    for n in raw {
        if *n == 0_u8 || *n == 0x0a_u8 || *n == 0x0d_u8 {
        //if *n < 0x20_u8 || *n > 0x7e_u8 {
           if buff.len() >= 5 {
               spl.push(buff.clone());
           }
           buff.clear();
        } else {
            buff.push((*n as f64) / 255_f64);
        }
    }

    spl
}

fn split_bytes(raw: &Vec<u8>) -> Vec<Vec<u8>> {
    let mut spl: Vec<Vec<u8>> = Vec::new();
    let mut buff: Vec<u8> = Vec::new();

    for n in raw {
        if *n == 0_u8 || *n == 0x0a_u8 || *n == 0x0d_u8 {
        //if *n < 0x20_u8 || *n > 0x7e_u8 { 
           if buff.len() >= 5 {
               spl.push(buff.clone());
           }
           buff.clear();
        } else {
            buff.push(*n);
        }
    }

    spl
}


fn first_15(s: &Vec<f64>) -> Vec<f64> {
    let mut v:Vec<f64> = Vec::new();
    let default_value = 0_f64;

    v.push(s.len() as f64 / 100.0);
    
    if s.len() >= 15 {
        for i in 0..15 {
            v.push(s[i]);
        }
    } else {
        for i in 0..s.len() {
            v.push(s[i]);
        }
        for _ in s.len()..15 {
            v.push(default_value);
        }
    }

    v
}


fn prepare_dataset(strs: Vec<Vec<f64>>, score: f64, data: &mut DataSet) {
    println!("preparing dataset with {} entries", strs.len());
    for s in strs {
        let v = first_15(&s);
        //println!("{:?}", v);
        if v.len() == 16 {
            data.push(&v, &[score]);
        }
    }
}
    

/*fn print_bytes(bs: Vec<u8>) {
    return match String::from_utf8(bs) {
        Ok(v) => println!("{}", v),
        Err(_) => (),
    }
}*/

fn prepare_crap(score: f64, data: &mut DataSet, amount: i32) {
    for i in 0..255 {
        let f = i as f64 / 255_f64;
        let mut v:Vec<f64> = Vec::new();
        v.push(0.0);
        for _ in 0..15 {
            v.push(f);
        }
        data.push(&v, &[score]);
    }

    let mut rng = rand::thread_rng();

    for _ in 0..amount {
        let mut v:Vec<f64> = Vec::new();
        let len = rng.gen_range(0..40);
        //v.push(len as f64);

        for _ in 0..len {
            v.push(rng.gen_range(0..256) as f64 / 255.0);
        }

        data.push(&first_15(&v), &[score]);
    }
}


fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        usage();
    }

    let mode = &args[1];

    if mode == "train" {
        let good_file = &args[2];
        let model_file = &args[3];

        println!("parsing ...");

        let mut data = DataSet::new();

        let good_raw = load(good_file.as_str());
        let good_spl = split_bytes_float(&good_raw);

        let crap_raw = load("crap.bin");
        let crap_spl = split_bytes_float(&crap_raw);



        println!("total dataset {} entries", crap_spl.len()+good_spl.len()+100_000);
        prepare_crap(-1_f64, &mut data, 100_000);
        prepare_dataset(crap_spl, -1_f64, &mut data);
        prepare_dataset(good_spl, 6_f64, &mut data);

        let mut nn = FeedForward::new(&[16,20,100,255,255,100,50,10,3,1]);

        println!("training ...");
        nn.activation(Tanh).learning_rate(0.0001).train(&data, 500_000);
        neuroflow::io::save(&nn, model_file).expect("cannot save the trained model");
        println!("saved {}", model_file);

    } else if mode == "exec" {
        let net = &args[2];
        let file = &args[3];

        let mut nn: FeedForward = neuroflow::io::load(net).expect("cannot load the trained model");

        let raw = load(file.as_str());
        let spl = split_bytes_float(&raw);
        let spl2 = split_bytes(&raw);

        for i in 0..spl.len() {
            let s = spl[i].clone(); 
            let f15 = first_15(&s);
            let r = nn.calc(&f15)[0];
            
            //print_bytes(spl2[i].clone());
            /* 
            match String::from_utf8(spl2[i].clone()) {
                Ok(v) => println!("{} {}", v, r),
                Err(_) => (),
            }*/

            if r >= 3.9 {
                match String::from_utf8(spl2[i].clone()) {  
                    Ok(v) => {
                        if v.len() > 4 {
                            //println!("{}  rate:{} {:?}", v, r, spl2[i]);
                            //println!("{}  rate:{}", v, r);
                            println!("{}", v);
                        }
                    },
                    Err(_) => (),
                };
            }
        }
        println!("---");
    } else {
        usage();
    }
}

