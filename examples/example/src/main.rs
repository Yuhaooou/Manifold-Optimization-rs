use manifold_optimization::algorithm::{BackTrackingParams, RGD, RTR};
use manifold_optimization::manifolds::*;
use manifold_optimization::problem::{FuncGradHess, Problem};
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

    let func = FuncGradHess::new(manifold.clone(), {
        move |x: &Array2<f64>, v: Option<&Array2<f64>>, c1, c2, c3| {
            let ax = mat.dot(x);
            let cost = if c1 { Some(0.5 * x.inner(&ax)) } else { None };
            let grad = if c2 {
                Some(manifold.egrad_to_rgrad(x, &ax))
            } else {
                None
            };
            let hess = if c3 {
                let v = v.unwrap();
                Some(manifold.ehess_to_rhess(x, v, &ax, &mat.dot(v)))
            } else {
                None
            };
            (cost, grad, hess)
        }
    });

    let problem = Problem::new_with_rng(func, &mut rng);

    // ==========================RGD=============================================

    println!("\n{:=^80}", "RGD");
    let start_time = std::time::Instant::now();

    let linesearch_params = BackTrackingParams::new(0.5, 0.8);
    let mut rgd = RGD::new(problem, linesearch_params)
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

    let mut rtr = RTR::new(rgd.problem_move(), 10.0, 0.1)
        .set_max_iterations(1000)
        .set_verbose(0)
        .set_min_grad_norm(1e-6)
        .set_min_step_size(1e-9);

    let res = rtr.run(1.);

    let end_time = std::time::Instant::now();

    println!("{res}");
    println!("  Time used: {:.2?}", end_time - start_time);
}
