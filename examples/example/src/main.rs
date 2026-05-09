use manifold_optimization::algorithm::{BackTrackingParams, RGD, RTR};
use manifold_optimization::manifolds::*;
use manifold_optimization::problem::Problem;
use manifold_optimization::utils::traits::InnerProduct;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use rand::prelude::*;

pub fn main() {
    let n: usize = 50;
    let r: usize = 30;

    // let mut rng = SmallRng::seed_from_u64(2026_04_05);
    let mut rng = SmallRng::from_os_rng();
    let mat_a = Array2::random_using((n, n), Normal::new(0., 1.).unwrap(), &mut rng);
    let mat = &mat_a + &mat_a.t();

    let manifold = Stiefel::new(n, r);

    let mut problem = Problem::new_with_rng(&manifold, |x| 0.5 * x.inner(&mat.dot(x)), &mut rng)
        .with_egrad_ehess(|x| mat.dot(x), |_, v| mat.dot(v));

    // ==========================RGD=============================================

    println!("\n{:=^80}", "RGD");
    let start_time = std::time::Instant::now();

    let linesearch_params = BackTrackingParams::new(0.5, 0.8);
    let mut rgd = RGD::new(&mut problem, &linesearch_params)
        .set_verbose(1)
        .set_max_iterations(1000)
        .set_min_grad_norm(1e-6)
        .set_min_step_size(1e-9);

    let res = rgd.run();

    let end_time = std::time::Instant::now();

    println!("{}", res);
    println!("  Time used: {:.2?}", end_time - start_time);

    // ===========================RTR=============================================

    println!("\n{:=^80}", "RTR");
    let start_time = std::time::Instant::now();

    let mut rtr = RTR::new(&mut problem, 10.0, 0.1)
        .set_max_iterations(1000)
        .set_verbose(0)
        .set_min_grad_norm(1e-6)
        .set_min_step_size(1e-9);

    let res = rtr.run(1.);

    let end_time = std::time::Instant::now();

    println!("{res}");
    println!("  Time used: {:.2?}", end_time - start_time);
}
