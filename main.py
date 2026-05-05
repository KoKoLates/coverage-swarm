import argparse

from core import Env
from planner import Planner, PrioritizedPlanner, CBSPlanner
from placement import Coverage, ParticleSwarmOptimizer, GA
from visualizer import SignalVisualizer


COV_OPT_REGISTRY: dict[str, Coverage] = {
    "ga":  GA,
    "pso": ParticleSwarmOptimizer,
}

PLANNER_REGISTRY: dict[str, Planner] = {
    "prioritized": lambda env: PrioritizedPlanner(env, "far"),
    "cbs":         lambda env: CBSPlanner(env),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coverage path planning pipeline")
    parser.add_argument("-f", "--file",    type=str, required=True,
                        help="filepath to map file")
    parser.add_argument("-c", "--cov-opt", type=str, default="ga",
                        choices=COV_OPT_REGISTRY.keys(),
                        help="coverage optimizer: ga | pso  (default: ga)")
    parser.add_argument("-p", "--planner", type=str, default="prioritized",
                        choices=PLANNER_REGISTRY.keys(),
                        help="path planner: prioritized | cbs  (default: prioritized)")
    parser.add_argument("-o", "--output",  type=str, default=None,
                        help="(optional) output file path for animation")
    args = parser.parse_args()

    env: Env            = Env(args.file)
    cov_opt: Coverage   = COV_OPT_REGISTRY[args.cov_opt](env)
    points              = cov_opt.process()
    planner: Planner    = PLANNER_REGISTRY[args.planner](env)
    paths               = planner.process(points)

    viz = SignalVisualizer(env)
    viz.animate(paths, args.output)
