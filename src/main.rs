// #![allow(unused)]
use ndarray::prelude::*;
use ndarray_rand::{RandomExt, rand_distr::Normal};
use rand::prelude::*;

use manifold_optimization::algorithm::RTR;
use manifold_optimization::manifolds::Sphere;
use manifold_optimization::problem::{EGradient, EHessian, Function, Problem};
use manifold_optimization::utils::random_point::RandomOn;

#[derive(Debug, Clone)]
struct Fun {
    mat: Array2<f64>,
}

impl Function for Fun {
    type Point = Array1<f64>;
    type Field = f64;

    fn value(&self, x: &Self::Point) -> f64 {
        0.5 * x.dot(&self.mat.dot(x))
    }
}

impl EGradient for Fun {
    type Point = Array1<f64>;
    type Gradient = Array1<f64>;

    fn euclidean_gradient(&self, x: &Self::Point) -> Self::Gradient {
        self.mat.dot(x)
    }
}

impl EHessian for Fun {
    type Point = Array1<f64>;
    type Direction = Array1<f64>;
    type Hessian = Array1<f64>;

    fn euclidean_hessian(&self, _x: &Self::Point, u: &Self::Direction) -> Self::Hessian {
        self.mat.dot(u)
    }
}

const R: usize = 5000;

fn main() {
    let mut rng = SmallRng::from_os_rng();
    let mat_a = Array2::random_using((R, R), Normal::new(0., 1.).unwrap(), &mut rng);
    let mat = &mat_a + &mat_a.t();

    let manifold = Sphere::new(R);
    let function = Fun { mat };
    let init_point = manifold.random_point();

    let mut problem = Problem::new(&manifold, &function).set_initial_point(init_point);

    // println!("============RGD============");
    // let linesearch_params = BackTrackingParams::new(0.1, 0.9);
    // let mut rgd = RGD::new(&mut problem, &linesearch_params);

    // let res = rgd.run();

    // println!("RGD converged in {} iterations", res.iters);
    // println!("Optimal cost: {:.8}", res.final_value);
    // println!("Final grad: {:.6e}", res.final_grad_norm);

    println!("\n{:=^80}", "RTR");
    let start_time = std::time::Instant::now();

    let mut rtr = RTR::new(&mut problem, 10.0, 0.1)
        .set_max_iterations(10000)
        .set_min_grad_norm(1e-8)
        .set_min_step_size(1e-12);

    let res = rtr.run(5.);

    let end_time = std::time::Instant::now();

    println!("{res}");
    println!("  Time used: {:.2?}s", end_time - start_time);
}
