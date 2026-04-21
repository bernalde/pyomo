# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Shared helpers for MindtPy unit tests."""

import logging
import time
from types import SimpleNamespace

from pyomo.common.timing import TicTocTimer
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_GOA_config
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_OA_config
from pyomo.core import Binary
from pyomo.core import Block
from pyomo.core import ConcreteModel
from pyomo.core import Constraint
from pyomo.core import ConstraintList
from pyomo.core import NonNegativeReals
from pyomo.core import Objective
from pyomo.core import Reals
from pyomo.core import Suffix
from pyomo.core import Var
from pyomo.core import VarList
from pyomo.core import minimize
from pyomo.opt import SolverResults
from pyomo.opt import SolverStatus
from pyomo.opt import TerminationCondition as tc


def make_logger(name='pyomo.contrib.mindtpy.tests'):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    return logger


def make_config(factory=_get_MindtPy_OA_config, **overrides):
    config = factory()
    config.logger = make_logger(factory.__name__)
    config.use_fbbt = False
    config.call_before_subproblem_solve = lambda model: None
    config.call_after_subproblem_solve = lambda model: None
    config.call_after_main_solve = lambda model: None
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


class FakePersistentBase:
    pass


class FakeSolver:
    def __init__(
        self,
        available=True,
        licensed=True,
        version=(1, 7, 0),
        solve_result=None,
    ):
        self._available = available
        self._licensed = licensed
        self._version = version
        self.solve_result = solve_result or make_results()
        self.solve_calls = []
        self.instance = None
        self.options = {}
        self.config = SimpleNamespace(options={}, time_limit=None, mip_gap=None)
        self.update_config = SimpleNamespace()
        self.callback = None

    def available(self):
        return self._available

    def license_is_valid(self):
        return self._licensed

    def version(self):
        return self._version

    def solve(self, *args, **kwargs):
        self.solve_calls.append((args, kwargs))
        return self.solve_result

    def set_instance(self, instance, *args, **kwargs):
        self.instance = instance
        self.instance_args = args
        self.instance_kwargs = kwargs

    def set_callback(self, callback):
        self.callback = callback


class FakePersistentSolver(FakeSolver, FakePersistentBase):
    pass


class FakeCallbackSolverModel:
    def __init__(self):
        self.warning_stream = None
        self.log_stream = None
        self.error_stream = None
        self.callback = None

    def register_callback(self, callback_cls):
        self.callback = SimpleNamespace(callback_cls=callback_cls)
        return self.callback

    def set_warning_stream(self, stream):
        self.warning_stream = stream

    def set_log_stream(self, stream):
        self.log_stream = stream

    def set_error_stream(self, stream):
        self.error_stream = stream


def make_results(
    termination=tc.optimal,
    status=SolverStatus.ok,
    lower_bound=None,
    upper_bound=None,
    message=None,
):
    results = SolverResults()
    results.solver.status = status
    results.solver.termination_condition = termination
    results.solver.message = message
    if lower_bound is not None:
        results.problem.lower_bound = lower_bound
    if upper_bound is not None:
        results.problem.upper_bound = upper_bound
    return results


def make_algorithm(model=None, factory=_get_MindtPy_OA_config, **config_overrides):
    algorithm = _MindtPyAlgorithm()
    algorithm.config = make_config(factory, **config_overrides)
    algorithm.results = SolverResults()
    algorithm.timing = TicTocTimer()
    algorithm.best_solution_found = None
    algorithm.best_solution_found_time = None
    algorithm.primal_integral = 0
    algorithm.dual_integral = 0
    algorithm.primal_dual_gap_integral = 0
    algorithm.nlp_infeasible_counter = 0
    algorithm.integer_list = []
    algorithm.curr_int_sol = None
    algorithm.last_iter_cuts = True
    algorithm.should_terminate = False
    algorithm.mip_iter = 0
    algorithm.nlp_iter = 0
    algorithm.fp_iter = 0
    algorithm.primal_bound_improved = False
    algorithm.dual_bound_improved = False
    algorithm.primal_bound_progress_time = []
    algorithm.dual_bound_progress_time = []
    algorithm.timing.main_timer_start_time = time.time()
    if model is not None:
        algorithm.set_up_solve_data(model)
        algorithm.create_utility_block(algorithm.working_model, 'MindtPy_utils')
        algorithm.objective_sense = next(
            algorithm.working_model.component_data_objects(Objective, active=True)
        ).sense
    else:
        algorithm.objective_sense = minimize
    return algorithm


def make_core_model(with_binary=True, nonlinear=True, sense=minimize):
    model = ConcreteModel()
    model.x = Var(bounds=(-2, 3), initialize=1.0)
    if with_binary:
        model.y = Var(domain=Binary, initialize=1)
    else:
        model.y = Var(bounds=(0, 2), initialize=0.5)
    if nonlinear:
        model.c = Constraint(expr=model.x**2 + model.y <= 4)
        model.obj = Objective(expr=model.x**2 + model.y, sense=sense)
    else:
        model.c = Constraint(expr=model.x + model.y <= 4)
        model.obj = Objective(expr=model.x + model.y, sense=sense)
    return model


def make_cut_model(
    include_binary=True,
    equality=False,
    lower=None,
    upper=None,
    sense=minimize,
):
    model = ConcreteModel()
    model.x = Var(bounds=(-2, 3), initialize=1.0)
    if include_binary:
        model.y = Var(domain=Binary, initialize=1)
        discrete_vars = [model.y]
    else:
        model.y = Var(bounds=(0, 2), initialize=0.5)
        discrete_vars = []
    body = model.x**2 + model.y
    if equality:
        model.c = Constraint(expr=body == (upper if upper is not None else 2))
    elif lower is not None and upper is not None:
        model.c = Constraint(expr=(lower, body, upper))
    elif upper is not None:
        model.c = Constraint(expr=body <= upper)
    else:
        model.c = Constraint(expr=body <= 2.0)
    model.obj = Objective(expr=body, sense=sense)
    model.MindtPy_utils = Block()
    model.MindtPy_utils.variable_list = [model.x, model.y]
    model.MindtPy_utils.discrete_variable_list = discrete_vars
    model.MindtPy_utils.continuous_variable_list = [model.x]
    model.MindtPy_utils.constraint_list = [model.c]
    model.MindtPy_utils.nonlinear_constraint_list = [model.c]
    model.MindtPy_utils.objective_list = [model.obj]
    model.MindtPy_utils.grey_box_list = []
    model.MindtPy_utils.cuts = Block()
    model.MindtPy_utils.cuts.oa_cuts = ConstraintList()
    model.MindtPy_utils.cuts.ecp_cuts = ConstraintList()
    model.MindtPy_utils.cuts.no_good_cuts = ConstraintList()
    model.MindtPy_utils.cuts.aff_cuts = ConstraintList()
    model.MindtPy_utils.cuts.fp_orthogonality_cuts = ConstraintList()
    model.MindtPy_utils.cuts.slack_vars = VarList(domain=NonNegativeReals)
    model.MindtPy_utils.feas_opt = Block()
    model.MindtPy_utils.feas_opt.feas_constraints = ConstraintList()
    model.MindtPy_utils.feas_opt.slack_var = VarList(domain=NonNegativeReals)
    model.dual = Suffix(direction=Suffix.IMPORT)
    return model


def make_goa_config(**overrides):
    return make_config(_get_MindtPy_GOA_config, **overrides)
