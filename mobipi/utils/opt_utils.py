"""
Utilities related to sampling-based optimization with BO.

@yjy0625
"""
import os
import numpy as np
from tqdm import tqdm
from skopt import Optimizer
from skopt.space import Real
import cma

import plotly.graph_objects as go


def normalize(x, bounds):
    """
    Normalize data to [0, 1] range.
    
    Parameters:
    - x (np.ndarray): Input data.
    - bounds (list of tuples): [(min_x, max_x), (min_y, max_y), ...].
    
    Returns:
    - normalized_x (np.ndarray): Normalized data.
    """
    return (x - np.array([b[0] for b in bounds])) / np.array([b[1] - b[0] for b in bounds])


def denormalize(x, bounds):
    """
    Denormalize data from [0, 1] range back to original bounds.
    
    Parameters:
    - x (np.ndarray): Normalized data.
    - bounds (list of tuples): [(min_x, max_x), (min_y, max_y), ...].
    
    Returns:
    - original_x (np.ndarray): Data in original scale.
    """
    return x * np.array([b[1] - b[0] for b in bounds]) + np.array([b[0] for b in bounds])


def optimize_pose_batch(F, bounds, algorithm='bayesian', n_iterations=100, batch_size=10, 
                        normalize_data=True, track_info=True, initial_samples=None, seed=0,
                        print_fn=lambda a: a, log_file=None, **kwargs):
    """
    Optimize the robot base pose (x, y, a) using batch evaluation with optional data normalization.
    
    Parameters:
    - F (function): The function to optimize. Takes a numpy array of poses and returns a list of scores.
    - bounds (list of tuples): The bounds for the optimization, [(min_x, max_x), (min_y, max_y), (min_a, max_a)].
    - algorithm (str): The optimization algorithm to use, either 'bayesian' or 'cmaes'.
    - n_iterations (int): Number of iterations for the optimization.
    - batch_size (int): Number of samples to evaluate in each batch.
    - normalize_data (bool): Whether to normalize data to [0, 1] internally.
    - track_info (bool): Whether to track intermediate optimization data for visualization.
    - initial_samples (array-like): A set of initial samples to start the BO optimization with.
    - log_file (str): Path to the log file for detailed timing information. If None, logging is disabled.
    - **kwargs: Algorithm-specific arguments for customization.
    
    Returns:
    - best_pose (np.ndarray): The optimized pose (x, y, a).
    - history (dict): A dictionary containing 'sampled_points', 'sampled_scores', and 'predicted_scores' (if track_info=True).
    """
    # Setup detailed logging
    import os
    import time as time_module
    from datetime import datetime
    
    # Use provided log_file or default
    if log_file is None:
        log_file = "debug/time_calculation.txt"
    
    details_log_file = log_file
    os.makedirs(os.path.dirname(details_log_file), exist_ok=True)
    
    # Write header with function call parameters
    with open(details_log_file, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"optimize_pose_batch called at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
        f.write("\n[PARAMETERS]\n")
        f.write(f"  algorithm: {algorithm}\n")
        f.write(f"  bounds: {bounds}\n")
        f.write(f"  n_iterations: {n_iterations}\n")
        f.write(f"  batch_size: {batch_size}\n")
        f.write(f"  normalize_data: {normalize_data}\n")
        f.write(f"  track_info: {track_info}\n")
        f.write(f"  initial_samples: {len(initial_samples) if initial_samples is not None else 0} samples\n")
        f.write(f"  seed: {seed}\n")
        f.write(f"  kwargs: {kwargs}\n")
        f.write(f"  log_file: {details_log_file}\n")
        f.write("\n")
        f.flush()  # Ensure immediate write to disk
    
    function_start_time = time_module.time()
    
    history = {
        'sampled_points': [], 'sampled_scores': [],
        'predicted_scores': [], 'rendered_images': [],
        'batch_size': batch_size,
        'n_iterations': n_iterations,
    } if track_info else None

    # Normalize bounds for [0, 1] range if requested
    if normalize_data:
        original_bounds = bounds
        bounds = [(0, 1)] * len(bounds)

        def normalized_F(poses):
            denormalized_poses = denormalize(poses, original_bounds)
            if len(denormalized_poses) < batch_size:
                return F(denormalized_poses)
            else:
                num_batches = len(denormalized_poses) // batch_size
                all_scores, all_info = [], {}
                for i in range(num_batches):
                    scores, info = F(denormalized_poses[i * batch_size:(i + 1) * batch_size])
                    all_scores.extend(scores)
                    if i == 0:
                        all_info = info
                    else:
                        for k in all_info.keys():
                            all_info[k] += info[k]
                return all_scores, all_info
    else:
        original_bounds = bounds
        normalized_F = F

    # Clip samples to bounds
    def clip_to_bounds(samples):
        return np.clip(samples, [b[0] for b in bounds], [b[1] for b in bounds])

    # Convert bounds into skopt format
    space = [Real(b[0], b[1], name=f'x{i}') for i, b in enumerate(bounds)]

    def objective_batch(X):
        X_clipped = clip_to_bounds(X)
        scores, score_info = normalized_F(np.array(X_clipped))
        if track_info:
            history['sampled_points'].extend(X_clipped)
            history['sampled_scores'].extend(scores)
            if "rendered_images" in score_info:
                history['rendered_images'].extend(score_info['rendered_images'])
        # we labelled -1, -2 scores for logging purposes
        # all scores sent to BO should be non-negative
        # so costs should be all negative
        cost = [-max(0, s) for s in scores]
        return cost

    if algorithm == 'bayesian':
        base_estimator = kwargs.get('base_estimator', 'GP')
        acq_func = kwargs.get('acq_func', 'EI')
        acq_func_kwargs = None
        use_dynamic_kappa = False
        if "acq_func_kwargs" in kwargs and "kappa" in kwargs["acq_func_kwargs"]:
            if type(kwargs["acq_func_kwargs"]["kappa"]) is float:
                acq_func_kwargs = dict(kappa=kwargs["acq_func_kwargs"]["kappa"])
            else:
                acq_func_kwargs = dict(kappa=kwargs["acq_func_kwargs"]["kappa"][0])
                kappa_start, kappa_end = kwargs["acq_func_kwargs"]["kappa"]
                alpha = np.log(kappa_start / kappa_end) / n_iterations
                def dynamic_kappa(t):
                    return kappa_start * np.exp(-alpha * t)
                use_dynamic_kappa = True
        opt = Optimizer(dimensions=space, base_estimator=base_estimator, acq_func=acq_func,
                        acq_func_kwargs=acq_func_kwargs,
                        n_jobs=-1, random_state=seed)

        # Add initial samples if provided
        if initial_samples is not None:
            import time
            initial_samples_clipped = clip_to_bounds(np.array(initial_samples))
            
            with open(details_log_file, 'a') as f:
                f.write("[INITIALIZATION PHASE]\n")
                f.write(f"  Number of initial samples: {len(initial_samples)}\n")
                f.flush()
            
            print(f"[BO] Evaluating {len(initial_samples)} initial samples...")
            eval_start = time.time()
            initial_costs = objective_batch(initial_samples_clipped)
            eval_time = time.time() - eval_start
            print(f"[BO] Initial sample evaluation complete! Time: {eval_time:.1f}s ({eval_time/60:.1f} min)")
            
            with open(details_log_file, 'a') as f:
                f.write(f"  Initial sample evaluation time: {eval_time:.2f}s ({eval_time/60:.2f} min)\n")
                f.write(f"    - Average time per sample: {eval_time/len(initial_samples):.3f}s\n")
                f.flush()
            
            print(f"[BO] Training Gaussian Process model with {len(initial_samples)} samples (this may take 5-15 minutes)...")
            gp_start = time.time()
            opt.tell(initial_samples_clipped.tolist(), initial_costs)
            gp_time = time.time() - gp_start
            print(f"[BO] GP model training complete! Time: {gp_time:.1f}s ({gp_time/60:.1f} min)")
            print(f"[BO] Total initialization time: {(eval_time + gp_time)/60:.1f} min")
            
            with open(details_log_file, 'a') as f:
                f.write(f"  GP model training time: {gp_time:.2f}s ({gp_time/60:.2f} min)\n")
                f.write(f"  Total initialization time: {(eval_time + gp_time):.2f}s ({(eval_time + gp_time)/60:.2f} min)\n")
                f.write(f"  Initial costs stats: min={np.min(initial_costs):.4f}, max={np.max(initial_costs):.4f}, mean={np.mean(initial_costs):.4f}\n")
                f.write("\n")
                f.flush()

        print(f"[BO] Starting {n_iterations} BO iterations with batch_size={batch_size}...")
        bo_start_time = time.time()
        
        with open(details_log_file, 'a') as f:
            f.write("[BO ITERATION PHASE]\n")
            f.write(f"  Number of iterations: {n_iterations}\n")
            f.write(f"  Batch size: {batch_size}\n")
            f.write("\n")
            f.flush()
        
        pbar = tqdm(range(n_iterations))
        for t in pbar:
            iter_start = time.time()
            
            # Ask for next batch of points
            ask_start = time.time()
            batch_points = opt.ask(n_points=batch_size)
            ask_time = time.time() - ask_start
            
            # Evaluate batch
            batch_points_clipped = clip_to_bounds(batch_points)
            eval_start = time.time()
            batch_costs = objective_batch(batch_points_clipped)
            eval_time = time.time() - eval_start
            
            # Update GP model
            tell_start = time.time()
            opt.tell(batch_points_clipped.tolist(), batch_costs)
            tell_time = time.time() - tell_start
            
            iter_time = time.time() - iter_start

            if use_dynamic_kappa:
                opt.acq_func_kwargs["kappa"] = dynamic_kappa(t)
            
            # Log iteration details
            current_best_score = np.max(history['sampled_scores']) if track_info else 0
            with open(details_log_file, 'a') as f:
                f.write(f"  Iteration {t+1}/{n_iterations}:\n")
                f.write(f"    - Ask time: {ask_time:.3f}s\n")
                f.write(f"    - Eval time: {eval_time:.3f}s (avg {eval_time/batch_size:.3f}s per sample)\n")
                f.write(f"    - Tell time: {tell_time:.3f}s\n")
                f.write(f"    - Total iter time: {iter_time:.3f}s\n")
                f.write(f"    - Batch costs: min={np.min(batch_costs):.4f}, max={np.max(batch_costs):.4f}, mean={np.mean(batch_costs):.4f}\n")
                f.write(f"    - Current best score: {current_best_score:.4f}\n")
                f.write(f"    - Kappa: {opt.acq_func_kwargs['kappa']:.4f}\n")
                f.flush()  # Flush after each iteration for real-time updates
            
            # Store predicted scores for visualization
            if track_info and hasattr(opt.models[-1], "predict"):
                best_idx = np.argmax(history['sampled_scores'])
                best_pose = history['sampled_points'][best_idx]
                pbar.set_description(f"BO (Max Score: {np.max(history['sampled_scores']):.3f}, " + \
                                     f"Best Pose: {print_fn(denormalize(best_pose.round(2), original_bounds))}, " + \
                                     f"Kappa: {opt.acq_func_kwargs['kappa']:.2f}, " + \
                                     f"Iter: {iter_time:.1f}s [Ask:{ask_time:.1f}s Eval:{eval_time:.1f}s Tell:{tell_time:.1f}s])")

        bo_total_time = time.time() - bo_start_time
        print(f"[BO] BO iterations complete! Time: {bo_total_time:.1f}s ({bo_total_time/60:.1f} min)")
        
        with open(details_log_file, 'a') as f:
            f.write(f"\n  Total BO iteration time: {bo_total_time:.2f}s ({bo_total_time/60:.2f} min)\n")
            f.write(f"  Average time per iteration: {bo_total_time/n_iterations:.3f}s\n")
            f.flush()
        
        best_pose = np.array(opt.Xi[np.argmin(opt.yi)])
        history['predicted_scores'] = None

    else:
        raise ValueError("Unknown algorithm: {}".format(algorithm))

    # Denormalize results if normalization was applied
    if normalize_data:
        best_pose = denormalize(best_pose, original_bounds)
        if track_info:
            history['sampled_points'] = denormalize(np.array(history['sampled_points']), original_bounds)
    
    # Write final summary
    function_total_time = time_module.time() - function_start_time
    with open(details_log_file, 'a') as f:
        f.write("\n[FINAL SUMMARY]\n")
        f.write(f"  Best pose: {best_pose}\n")
        if track_info:
            f.write(f"  Best score: {np.max(history['sampled_scores']):.6f}\n")
            f.write(f"  Total samples evaluated: {len(history['sampled_scores'])}\n")
        f.write(f"  Total function time: {function_total_time:.2f}s ({function_total_time/60:.2f} min)\n")
        f.write("\n" + "="*80 + "\n\n")
        f.flush()

    return (best_pose, history) if track_info else best_pose
