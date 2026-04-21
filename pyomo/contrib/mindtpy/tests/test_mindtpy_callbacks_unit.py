# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap
from pyomo.contrib.mindtpy import single_tree
from pyomo.contrib.mindtpy import tabu_list
from pyomo.contrib.mindtpy.tests._helpers import make_config
from pyomo.contrib.mindtpy.tests._helpers import make_cut_model
from pyomo.contrib.mindtpy.tests._helpers import make_results
from pyomo.core import ConstraintList
from pyomo.core import minimize
from pyomo.core import value
from pyomo.opt import SolverResults
from pyomo.opt import TerminationCondition as tc


class TestTabuList(unittest.TestCase):
    def test_callback_helpers_extract_and_filter_integer_solutions(self):
        model = make_cut_model()
        solver_var_map = ComponentMap([(model.y, 'Y')])
        get_values = lambda key: {'Y': 0.6}[key]
        self.assertEqual(
            tabu_list._get_callback_integer_solution(
                model.MindtPy_utils.discrete_variable_list, solver_var_map, get_values
            ),
            (1,),
        )
        self.assertTrue(tabu_list._should_reject_incumbent(True, (1,), []))
        self.assertTrue(tabu_list._should_reject_incumbent(False, (1,), [(1,)]))
        self.assertFalse(tabu_list._should_reject_incumbent(False, (0,), [(1,)]))

    def test_callback_call_rejects_repeated_solution(self):
        model = make_cut_model()
        fake_callback = SimpleNamespace(
            mindtpy_solver=SimpleNamespace(mip=model, integer_list=[(1,)]),
            opt=SimpleNamespace(
                _pyomo_var_to_solver_var_map=ComponentMap([(model.y, 'Y')])
            ),
            config=SimpleNamespace(single_tree=False),
            get_values=lambda key: {'Y': 1.0}[key],
            reject=MagicMock(),
        )
        tabu_list.IncumbentCallback_cplex.__call__(fake_callback)
        self.assertEqual(fake_callback.mindtpy_solver.curr_int_sol, (1,))
        fake_callback.reject.assert_called_once()


class TestSingleTreeCallbacks(unittest.TestCase):
    def _fake_cplex(self):
        return SimpleNamespace(
            SparsePair=lambda ind, val: SimpleNamespace(variables=ind, coefficients=val),
            callbacks=SimpleNamespace(
                SolutionSource=SimpleNamespace(mipstart_solution=119)
            ),
        )

    def test_copy_values_and_add_lazy_no_good_cuts(self):
        callback = single_tree.LazyOACallback_cplex()
        source_model = make_cut_model()
        target_model = make_cut_model()
        opt = SimpleNamespace(
            _pyomo_var_to_solver_var_map=ComponentMap(
                [(source_model.x, 'x'), (source_model.y, 'y')]
            ),
            _get_expr_from_pyomo_expr=lambda expr: (
                SimpleNamespace(variables=['y'], coefficients=[1.0]),
                None,
            ),
        )
        callback.get_values = lambda key: {'x': 2.0, 'y': 0.0}[key]
        callback.copy_lazy_var_list_values(
            opt,
            source_model.MindtPy_utils.variable_list,
            target_model.MindtPy_utils.variable_list,
            make_config(),
            skip_fixed=False,
        )
        self.assertAlmostEqual(target_model.x.value, 2.0)
        self.assertEqual(target_model.y.value, 0)

        callback.add = MagicMock()
        with patch.object(single_tree, 'cplex', self._fake_cplex()):
            callback.add_lazy_no_good_cuts(
                [2.0, 0.0],
                SimpleNamespace(mip=target_model, timing={}),
                make_config(add_no_good_cuts=True),
                opt,
            )
        callback.add.assert_called_once()

    def test_add_lazy_oa_cuts_records_mipstart_cuts(self):
        callback = single_tree.LazyOACallback_cplex()
        model = make_cut_model(equality=True, upper=2.0)
        fake_solver = SimpleNamespace(
            jacobians=ComponentMap(
                [(model.c, ComponentMap([(model.x, 2.0), (model.y, 1.0)]))]
            ),
            mip_constraint_polynomial_degree={0, 1},
            objective_sense=minimize,
            mip_start_lazy_oa_cuts=[],
            timing={},
        )
        config = make_config()
        opt = SimpleNamespace(
            _pyomo_var_to_solver_var_map=ComponentMap(
                [(model.x, 'x'), (model.y, 'y')]
            ),
            _get_expr_from_pyomo_expr=lambda expr: (
                SimpleNamespace(variables=['x', 'y'], coefficients=[1.0, 1.0]),
                None,
            ),
        )
        callback.add = MagicMock()
        callback.get_values = lambda key: {'x': value(model.x), 'y': value(model.y)}[key]
        callback.get_solution_source = lambda: 119
        with patch.object(single_tree, 'cplex', self._fake_cplex()):
            callback.add_lazy_oa_cuts(
                model,
                [1.0],
                fake_solver,
                config,
                opt,
            )
        self.assertEqual(len(fake_solver.mip_start_lazy_oa_cuts), 1)
        callback.add.assert_called_once()

    def test_copy_values_skip_flags_and_inequality_oa_paths(self):
        callback = single_tree.LazyOACallback_cplex()
        source_model = make_cut_model(lower=2.0, upper=3.0)
        target_model = make_cut_model(lower=2.0, upper=3.0)
        source_model.x.stale = True
        target_model.y.fix(1)
        opt = SimpleNamespace(
            _pyomo_var_to_solver_var_map=ComponentMap(
                [(source_model.x, 'x'), (source_model.y, 'y')]
            ),
            _get_expr_from_pyomo_expr=lambda expr: (
                SimpleNamespace(variables=['x', 'y'], coefficients=[1.0, 1.0]),
                None,
            ),
        )
        callback.get_values = lambda key: {'x': 2.0, 'y': 0.0}[key]
        callback.copy_lazy_var_list_values(
            opt,
            source_model.MindtPy_utils.variable_list,
            target_model.MindtPy_utils.variable_list,
            make_config(),
            skip_stale=True,
            skip_fixed=True,
        )
        self.assertAlmostEqual(target_model.x.value, 1.0)
        self.assertEqual(target_model.y.value, 1)

        fake_solver = SimpleNamespace(
            jacobians=ComponentMap(
                [(source_model.c, ComponentMap([(source_model.x, 2.0), (source_model.y, 1.0)]))]
            ),
            mip_constraint_polynomial_degree={0, 1},
            objective_sense=minimize,
            mip_start_lazy_oa_cuts=[],
            timing={},
        )
        callback.add = MagicMock()
        callback.get_solution_source = lambda: 119
        with patch.object(single_tree, 'cplex', self._fake_cplex()):
            callback.add_lazy_oa_cuts(
                source_model,
                [1.0],
                fake_solver,
                make_config(linearize_inactive=True),
                opt,
            )
        self.assertEqual(callback.add.call_count, 2)
        self.assertEqual(len(fake_solver.mip_start_lazy_oa_cuts), 2)

    def test_add_lazy_affine_and_no_good_cut_edge_cases(self):
        callback = single_tree.LazyOACallback_cplex()
        model = make_cut_model(include_binary=False, upper=2.0)
        opt = SimpleNamespace(
            _get_expr_from_pyomo_expr=lambda expr: (
                SimpleNamespace(variables=['x', 'y'], coefficients=[1.0, 1.0]),
                None,
            )
        )
        callback.add = MagicMock()
        model.x.set_value(None, skip_validation=True)
        with patch.object(single_tree, 'cplex', self._fake_cplex()):
            callback.add_lazy_affine_cuts(
                SimpleNamespace(mip=model, timing={}), make_config(), opt
            )
        self.assertFalse(callback.add.called)

        model.x.set_value(1.0, skip_validation=True)

        class ErrorMcCormick:
            def __init__(self, _expr):
                raise single_tree.MCPP_Error('bad mcpp evaluation')

        class InfiniteMcCormick:
            def __init__(self, _expr):
                pass

            def subcc(self):
                return ComponentMap([(model.x, float('inf')), (model.y, 0.0)])

            def subcv(self):
                return ComponentMap([(model.x, float('inf')), (model.y, 0.0)])

            def concave(self):
                return float('inf')

            def convex(self):
                return float('inf')

            def upper(self):
                return 3.0

            def lower(self):
                return -1.0

        class ZeroMcCormick:
            def __init__(self, _expr):
                pass

            def subcc(self):
                return ComponentMap([(model.x, 0.0), (model.y, 0.0)])

            def subcv(self):
                return ComponentMap([(model.x, 0.0), (model.y, 0.0)])

            def concave(self):
                return 0.0

            def convex(self):
                return 0.0

            def upper(self):
                return 3.0

            def lower(self):
                return -1.0

        with patch.object(single_tree, 'mc', ErrorMcCormick), patch.object(
            single_tree, 'cplex', self._fake_cplex()
        ):
            callback.add_lazy_affine_cuts(
                SimpleNamespace(mip=model, timing={}), make_config(), opt
            )
        with patch.object(single_tree, 'mc', InfiniteMcCormick), patch.object(
            single_tree, 'cplex', self._fake_cplex()
        ):
            callback.add_lazy_affine_cuts(
                SimpleNamespace(mip=model, timing={}), make_config(), opt
            )
        with patch.object(single_tree, 'mc', ZeroMcCormick), patch.object(
            single_tree, 'cplex', self._fake_cplex()
        ):
            callback.add_lazy_affine_cuts(
                SimpleNamespace(mip=model, timing={}), make_config(), opt
            )
        self.assertFalse(callback.add.called)

        no_good_callback = single_tree.LazyOACallback_cplex()
        no_good_callback.add = MagicMock()
        with patch.object(single_tree, 'cplex', self._fake_cplex()):
            no_good_callback.add_lazy_no_good_cuts(
                [1.0, 0.0],
                SimpleNamespace(mip=make_cut_model(), timing={}),
                make_config(add_no_good_cuts=False),
                opt,
            )
            no_good_callback.add_lazy_no_good_cuts(
                [1.0, 0.5],
                SimpleNamespace(mip=make_cut_model(include_binary=False), timing={}),
                make_config(add_no_good_cuts=True),
                opt,
            )
        self.assertFalse(no_good_callback.add.called)

        with patch.object(single_tree, 'cplex', self._fake_cplex()):
            with self.assertRaisesRegex(ValueError, 'is not 0 or 1'):
                no_good_callback.add_lazy_no_good_cuts(
                    [1.0, 0.25],
                    SimpleNamespace(mip=make_cut_model(), timing={}),
                    make_config(add_no_good_cuts=True, integer_tolerance=1e-8),
                    opt,
                )

    def test_handle_lazy_subproblem_optimal_and_infeasible(self):
        callback = single_tree.LazyOACallback_cplex()
        fixed_nlp = make_cut_model()
        fixed_nlp.MindtPy_utils.objective_list = [fixed_nlp.obj]
        fixed_nlp.tmp_duals = fixed_nlp.MindtPy_utils.constraint_list
        fixed_nlp.dual[fixed_nlp.c] = 1.0
        solver = SimpleNamespace(
            mip=make_cut_model(),
            fixed_nlp_log_formatter='{0}{1}{2}{3}{4}{5}{6}',
            primal_bound=10.0,
            dual_bound=4.0,
            rel_gap=0.2,
            primal_bound_improved=False,
            best_solution_found=None,
            best_solution_found_time=None,
            stored_bound={},
            jacobians=ComponentMap(),
            objective_sense=minimize,
            mip_constraint_polynomial_degree={0, 1},
            mip_iter=1,
            timing=SimpleNamespace(),
            update_primal_bound=lambda bound: None,
            nlp_iter=1,
            solve_feasibility_subproblem=lambda: (make_cut_model(), SolverResults()),
            nlp_infeasible_counter=0,
        )
        solver.update_primal_bound = lambda bound: setattr(
            solver, 'primal_bound_improved', True
        )
        config = make_config(
            calculate_dual_at_solution=True,
            add_no_good_cuts=True,
            add_regularization='grad_lag',
            strategy='OA',
            nlp_solver='cyipopt',
        )
        with patch.object(single_tree, 'copy_var_list_values') as copy_values, patch.object(
            callback, 'add_lazy_oa_cuts'
        ) as add_lazy_oa, patch.object(
            callback, 'add_lazy_no_good_cuts'
        ) as add_nogood, patch.object(
            single_tree, 'add_oa_cuts'
        ) as add_oa_cuts, patch.object(
            single_tree, 'get_main_elapsed_time', return_value=1.0
        ):
            callback.handle_lazy_subproblem_optimal(
                fixed_nlp, solver, config, SimpleNamespace()
            )
        copy_values.assert_called()
        add_lazy_oa.assert_called_once()
        add_nogood.assert_called_once()
        add_oa_cuts.assert_called_once()
        self.assertIsNotNone(solver.best_solution_found)

        goa_config = make_config(strategy='GOA', add_no_good_cuts=True)
        with patch.object(single_tree, 'copy_var_list_values'), patch.object(
            callback, 'add_lazy_affine_cuts'
        ) as add_affine, patch.object(
            callback, 'add_lazy_no_good_cuts'
        ) as add_nogood:
            callback.handle_lazy_subproblem_infeasible(
                fixed_nlp, solver, goa_config, SimpleNamespace()
            )
        add_affine.assert_called_once()
        add_nogood.assert_called_once()

    def test_handle_lazy_subproblem_remaining_paths(self):
        callback = single_tree.LazyOACallback_cplex()
        fixed_nlp = make_cut_model()
        fixed_nlp.MindtPy_utils.objective_list = [fixed_nlp.obj]
        solver = SimpleNamespace(
            mip=make_cut_model(),
            fixed_nlp_log_formatter='{0}{1}{2}{3}{4}{5}{6}',
            primal_bound=10.0,
            dual_bound=4.0,
            rel_gap=0.2,
            primal_bound_improved=False,
            best_solution_found=None,
            best_solution_found_time=None,
            stored_bound={},
            jacobians=ComponentMap(),
            objective_sense=minimize,
            mip_constraint_polynomial_degree={0, 1},
            mip_iter=1,
            timing=SimpleNamespace(main_timer_start_time=0.0),
            nlp_iter=1,
            nlp_infeasible_counter=0,
            update_primal_bound=lambda bound: setattr(
                solver, 'primal_bound_improved', True
            ),
            solve_feasibility_subproblem=lambda: (make_cut_model(), SolverResults()),
        )
        with patch.object(single_tree, 'copy_var_list_values'), patch.object(
            callback, 'add_lazy_affine_cuts'
        ) as add_affine:
            callback.handle_lazy_subproblem_optimal(
                fixed_nlp,
                solver,
                make_config(
                    strategy='GOA',
                    calculate_dual_at_solution=False,
                    add_no_good_cuts=False,
                ),
                SimpleNamespace(),
            )
        add_affine.assert_called_once()

        fixed_nlp.dual[fixed_nlp.c] = None
        with patch.object(single_tree, 'copy_var_list_values'), patch.object(
            callback, 'add_lazy_oa_cuts'
        ) as add_lazy_oa, patch.object(
            callback, 'add_lazy_no_good_cuts'
        ) as add_nogood, patch.object(
            single_tree, 'add_oa_cuts'
        ) as add_oa_cuts:
            callback.handle_lazy_subproblem_infeasible(
                fixed_nlp,
                solver,
                make_config(
                    strategy='OA',
                    calculate_dual_at_solution=True,
                    add_no_good_cuts=True,
                    add_regularization='grad_lag',
                ),
                SimpleNamespace(),
            )
        add_lazy_oa.assert_called_once()
        add_oa_cuts.assert_called_once()
        add_nogood.assert_called_once()

    def test_add_lazy_affine_cuts_and_callback_branches(self):
        callback = single_tree.LazyOACallback_cplex()
        model = make_cut_model(include_binary=False, upper=2.0)
        callback.add = MagicMock()
        opt = SimpleNamespace(
            _get_expr_from_pyomo_expr=lambda expr: (
                SimpleNamespace(variables=['x', 'y'], coefficients=[1.0, 1.0]),
                None,
            )
        )

        class FakeMcCormick:
            def __init__(self, _expr):
                pass

            def subcc(self):
                return ComponentMap([(model.x, 1.5), (model.y, 0.5)])

            def subcv(self):
                return ComponentMap([(model.x, 1.0), (model.y, 0.25)])

            def concave(self):
                return 0.5

            def convex(self):
                return 0.75

            def upper(self):
                return 3.0

            def lower(self):
                return -1.0

        with patch.object(single_tree, 'mc', FakeMcCormick), patch.object(
            single_tree, 'cplex', self._fake_cplex()
        ):
            callback.add_lazy_affine_cuts(
                SimpleNamespace(mip=model, timing={}), make_config(), opt
            )
        self.assertEqual(callback.add.call_count, 2)

        repeated_callback = single_tree.LazyOACallback_cplex()
        repeated_callback.main_mip = make_cut_model()
        repeated_callback.config = make_config(
            strategy='GOA', add_no_good_cuts=True, add_cuts_at_incumbent=False
        )
        repeated_callback.opt = SimpleNamespace()
        repeated_callback.get_solution_source = lambda: 111
        repeated_callback.handle_lazy_main_feasible_solution = MagicMock()
        repeated_callback.add_lazy_no_good_cuts = MagicMock()
        repeated_callback.mindtpy_solver = SimpleNamespace(
            mip_start_lazy_oa_cuts=[],
            should_terminate=False,
            primal_bound=10.0,
            dual_bound=0.0,
            fixed_nlp=make_cut_model(),
            working_model=make_cut_model(),
            integer_list=[(1,)],
            results=SolverResults(),
            timing={},
            primal_bound_improved=False,
            dual_bound_improved=False,
            best_solution_found=None,
        )
        with patch.object(single_tree, 'cplex', self._fake_cplex()), patch.object(
            single_tree, 'get_integer_solution', return_value=(1,)
        ):
            repeated_callback()
        repeated_callback.add_lazy_no_good_cuts.assert_called_once()

        new_callback = single_tree.LazyOACallback_cplex()
        new_callback.main_mip = make_cut_model()
        new_callback.config = make_config(
            strategy='OA', add_no_good_cuts=False, add_cuts_at_incumbent=False
        )
        new_callback.opt = SimpleNamespace()
        new_callback.get_solution_source = lambda: 111
        new_callback.handle_lazy_main_feasible_solution = MagicMock()
        new_callback.handle_lazy_subproblem_optimal = MagicMock()
        new_callback.mindtpy_solver = SimpleNamespace(
            mip_start_lazy_oa_cuts=[],
            should_terminate=False,
            primal_bound=10.0,
            dual_bound=0.0,
            fixed_nlp=make_cut_model(),
            working_model=make_cut_model(),
            integer_list=[],
            results=SolverResults(),
            timing={},
            primal_bound_improved=False,
            dual_bound_improved=False,
            best_solution_found=None,
            solve_subproblem=lambda: (make_cut_model(), make_results()),
        )
        with patch.object(single_tree, 'cplex', self._fake_cplex()), patch.object(
            single_tree, 'get_integer_solution', return_value=(1,)
        ):
            new_callback()
        new_callback.handle_lazy_subproblem_optimal.assert_called_once()

    def test_handle_other_termination_and_main_solution_copy(self):
        callback = single_tree.LazyOACallback_cplex()
        fixed_nlp = make_cut_model()
        config = make_config()
        solver = SimpleNamespace(
            fixed_nlp=make_cut_model(),
            update_dual_bound=MagicMock(),
            mip_iter=1,
            primal_bound=10.0,
            dual_bound=5.0,
            rel_gap=0.5,
            timing=SimpleNamespace(),
            log_formatter='{0}{1}{2}{3}{4}{5}{6}',
        )
        callback.get_best_objective_value = lambda: 2.0
        callback.get_objective_value = lambda: 3.0
        callback.copy_lazy_var_list_values = MagicMock()
        with patch.object(single_tree, 'copy_var_list_values'), patch.object(
            single_tree, 'get_main_elapsed_time', return_value=1.0
        ):
            callback.handle_lazy_main_feasible_solution(
                make_cut_model(), solver, config, SimpleNamespace()
            )
        solver.update_dual_bound.assert_called_once_with(2.0)

        callback.handle_lazy_subproblem_other_termination(
            fixed_nlp, tc.maxIterations, solver, config
        )
        with self.assertRaisesRegex(ValueError, 'unable to handle NLP subproblem termination'):
            callback.handle_lazy_subproblem_other_termination(
                fixed_nlp, tc.error, solver, config
            )

    def test_lazy_callback_short_circuits_and_replays_mip_start_cuts(self):
        callback = single_tree.LazyOACallback_cplex()
        callback.add = MagicMock()
        callback.abort = MagicMock()
        callback.get_solution_source = lambda: 111
        callback.main_mip = make_cut_model()
        callback.config = make_config()
        callback.opt = SimpleNamespace()
        callback.mindtpy_solver = SimpleNamespace(
            mip_start_lazy_oa_cuts=[['cut', 'L', 1.0]],
            should_terminate=True,
        )
        with patch.object(single_tree, 'cplex', self._fake_cplex()):
            callback()
        callback.add.assert_called_once()
        callback.abort.assert_called_once()

    def test_lazy_callback_covers_incumbent_cut_and_regularization_returns(self):
        callback = single_tree.LazyOACallback_cplex()
        callback.main_mip = make_cut_model()
        callback.config = make_config(
            strategy='OA', add_cuts_at_incumbent=True, add_regularization=None
        )
        callback.opt = SimpleNamespace()
        callback.get_solution_source = lambda: 111
        callback.copy_lazy_var_list_values = MagicMock()
        callback.handle_lazy_main_feasible_solution = MagicMock()
        callback.add_lazy_oa_cuts = MagicMock(side_effect=ValueError('bad start'))
        callback.mindtpy_solver = SimpleNamespace(
            mip_start_lazy_oa_cuts=[],
            should_terminate=False,
            mip=make_cut_model(),
            fixed_nlp=make_cut_model(),
            working_model=make_cut_model(),
            integer_list=[],
            results=SolverResults(),
            timing={},
            primal_bound=10.0,
            dual_bound=0.0,
            best_solution_found=None,
            primal_bound_improved=False,
            dual_bound_improved=False,
        )
        with patch.object(single_tree, 'cplex', self._fake_cplex()):
            callback()
        callback.add_lazy_oa_cuts.assert_called_once()

        skip_callback = single_tree.LazyOACallback_cplex()
        skip_callback.main_mip = make_cut_model()
        skip_callback.config = make_config(
            strategy='OA', add_cuts_at_incumbent=False, add_regularization='grad_lag'
        )
        skip_callback.opt = SimpleNamespace()
        skip_callback.get_solution_source = lambda: 111
        skip_callback.handle_lazy_main_feasible_solution = MagicMock()
        skip_callback.abort = MagicMock()
        skip_callback.mindtpy_solver = SimpleNamespace(
            mip_start_lazy_oa_cuts=[],
            should_terminate=False,
            best_solution_found=make_cut_model(),
            dual_bound_improved=False,
            primal_bound_improved=False,
            dual_bound_progress=[0.0],
            primal_bound=10.0,
            dual_bound=0.0,
            results=SolverResults(),
            fixed_nlp=make_cut_model(),
            working_model=make_cut_model(),
            integer_list=[],
            timing={},
        )
        with patch.object(single_tree, 'cplex', self._fake_cplex()):
            skip_callback()
        skip_callback.abort.assert_not_called()

        bound_callback = single_tree.LazyOACallback_cplex()
        bound_callback.main_mip = make_cut_model()
        bound_callback.config = make_config(
            strategy='OA', add_cuts_at_incumbent=False, add_regularization='grad_lag'
        )
        bound_callback.opt = SimpleNamespace()
        bound_callback.get_solution_source = lambda: 111
        bound_callback.handle_lazy_main_feasible_solution = MagicMock()
        bound_callback.abort = MagicMock()
        bound_callback.mindtpy_solver = SimpleNamespace(
            mip_start_lazy_oa_cuts=[],
            should_terminate=False,
            best_solution_found=make_cut_model(),
            dual_bound_improved=True,
            primal_bound_improved=False,
            dual_bound_progress=[0.0],
            add_regularization=MagicMock(),
            primal_bound=1.0,
            dual_bound=1.0,
            results=SolverResults(),
            fixed_nlp=make_cut_model(),
            working_model=make_cut_model(),
            integer_list=[],
            timing={},
        )
        with patch.object(single_tree, 'cplex', self._fake_cplex()):
            bound_callback()
        bound_callback.mindtpy_solver.add_regularization.assert_called_once()
        bound_callback.abort.assert_called_once()
        self.assertIs(
            bound_callback.mindtpy_solver.results.solver.termination_condition,
            tc.optimal,
        )

    def test_lazy_callback_covers_remaining_subproblem_termination_paths(self):
        repeated_callback = single_tree.LazyOACallback_cplex()
        repeated_callback.main_mip = make_cut_model()
        repeated_callback.config = make_config(
            strategy='OA', add_cuts_at_incumbent=False
        )
        repeated_callback.opt = SimpleNamespace()
        repeated_callback.get_solution_source = lambda: 111
        repeated_callback.handle_lazy_main_feasible_solution = MagicMock()
        repeated_callback.mindtpy_solver = SimpleNamespace(
            mip_start_lazy_oa_cuts=[],
            should_terminate=False,
            primal_bound=10.0,
            dual_bound=0.0,
            fixed_nlp=make_cut_model(),
            working_model=make_cut_model(),
            integer_list=[(1,)],
            results=SolverResults(),
            timing={},
            primal_bound_improved=False,
            dual_bound_improved=False,
            best_solution_found=None,
            solve_subproblem=MagicMock(),
        )
        with patch.object(single_tree, 'cplex', self._fake_cplex()), patch.object(
            single_tree, 'get_integer_solution', return_value=(1,)
        ):
            repeated_callback()
        repeated_callback.mindtpy_solver.solve_subproblem.assert_not_called()

        infeasible_callback = single_tree.LazyOACallback_cplex()
        infeasible_callback.main_mip = make_cut_model()
        infeasible_callback.config = make_config(
            strategy='OA', add_cuts_at_incumbent=False
        )
        infeasible_callback.opt = SimpleNamespace()
        infeasible_callback.get_solution_source = lambda: 111
        infeasible_callback.handle_lazy_main_feasible_solution = MagicMock()
        infeasible_callback.handle_lazy_subproblem_infeasible = MagicMock()
        infeasible_callback.mindtpy_solver = SimpleNamespace(
            mip_start_lazy_oa_cuts=[],
            should_terminate=False,
            primal_bound=10.0,
            dual_bound=0.0,
            fixed_nlp=make_cut_model(),
            working_model=make_cut_model(),
            integer_list=[],
            results=SolverResults(),
            timing={},
            primal_bound_improved=False,
            dual_bound_improved=False,
            best_solution_found=None,
            solve_subproblem=lambda: (
                make_cut_model(),
                make_results(termination=tc.infeasible),
            ),
        )
        with patch.object(single_tree, 'cplex', self._fake_cplex()), patch.object(
            single_tree, 'get_integer_solution', return_value=(1,)
        ):
            infeasible_callback()
        infeasible_callback.handle_lazy_subproblem_infeasible.assert_called_once()

        other_callback = single_tree.LazyOACallback_cplex()
        other_callback.main_mip = make_cut_model()
        other_callback.config = make_config(
            strategy='OA', add_cuts_at_incumbent=False
        )
        other_callback.opt = SimpleNamespace()
        other_callback.get_solution_source = lambda: 111
        other_callback.handle_lazy_main_feasible_solution = MagicMock()
        other_callback.handle_lazy_subproblem_other_termination = MagicMock()
        other_callback.mindtpy_solver = SimpleNamespace(
            mip_start_lazy_oa_cuts=[],
            should_terminate=False,
            primal_bound=10.0,
            dual_bound=0.0,
            fixed_nlp=make_cut_model(),
            working_model=make_cut_model(),
            integer_list=[],
            results=SolverResults(),
            timing={},
            primal_bound_improved=False,
            dual_bound_improved=False,
            best_solution_found=None,
            solve_subproblem=lambda: (
                make_cut_model(),
                make_results(termination=tc.maxIterations),
            ),
        )
        with patch.object(single_tree, 'cplex', self._fake_cplex()), patch.object(
            single_tree, 'get_integer_solution', return_value=(1,)
        ):
            other_callback()
        other_callback.handle_lazy_subproblem_other_termination.assert_called_once()

        optimal_callback = single_tree.LazyOACallback_cplex()
        optimal_callback.main_mip = make_cut_model()
        optimal_callback.config = make_config(
            strategy='OA', add_cuts_at_incumbent=False
        )
        optimal_callback.opt = SimpleNamespace()
        optimal_callback.get_solution_source = lambda: 111
        optimal_callback.handle_lazy_main_feasible_solution = MagicMock()
        optimal_callback.handle_lazy_subproblem_optimal = MagicMock(
            side_effect=lambda fixed_nlp, solver, config, opt: setattr(
                solver, 'primal_bound', solver.dual_bound
            )
        )
        optimal_callback.mindtpy_solver = SimpleNamespace(
            mip_start_lazy_oa_cuts=[],
            should_terminate=False,
            primal_bound=1.0,
            dual_bound=0.0,
            fixed_nlp=make_cut_model(),
            working_model=make_cut_model(),
            integer_list=[],
            results=SolverResults(),
            timing={},
            primal_bound_improved=False,
            dual_bound_improved=False,
            best_solution_found=None,
            solve_subproblem=lambda: (make_cut_model(), make_results()),
        )
        with patch.object(single_tree, 'cplex', self._fake_cplex()), patch.object(
            single_tree, 'get_integer_solution', return_value=(2,)
        ):
            optimal_callback()
        self.assertIs(
            optimal_callback.mindtpy_solver.results.solver.termination_condition,
            tc.optimal,
        )

    def test_lazy_gurobi_callback_replays_previous_cuts(self):
        fake_grb = SimpleNamespace(
            Callback=SimpleNamespace(MIPSOL=1, MIPSOL_OBJBND=2),
            Param=SimpleNamespace(SolutionNumber='SolutionNumber'),
        )
        cb_model = make_cut_model()
        cb_model.MindtPy_utils.cuts.oa_cuts.add(expr=cb_model.y >= 0)
        cb_model.MindtPy_utils.cuts.oa_cuts.add(expr=cb_model.y <= 1)
        cb_opt = SimpleNamespace(
            _solver_model=SimpleNamespace(terminate=MagicMock(), SolCount=2, PoolObjVal=0),
            cbGetSolution=MagicMock(),
            cbGet=lambda key: 2.0,
            cbLazy=MagicMock(),
        )
        mindtpy_solver = SimpleNamespace(
            should_terminate=False,
            fixed_nlp=make_cut_model(),
            mip=cb_model,
            update_dual_bound=MagicMock(),
            timing=SimpleNamespace(),
            primal_bound=10.0,
            dual_bound=5.0,
            rel_gap=0.5,
            log_formatter='{0}{1}{2}{3}{4}{5}{6}',
            best_solution_found=None,
            dual_bound_improved=True,
            primal_bound_improved=False,
            dual_bound_progress=[0.0],
            integer_list=[(1,)],
            integer_solution_to_cuts_index={(1,): [1, 2]},
            results=SolverResults(),
            mip_iter=1,
            bounds_converged=lambda: False,
            reached_time_limit=lambda: False,
        )
        config = make_config(strategy='OA')
        with patch.object(single_tree, 'gurobipy', SimpleNamespace(GRB=fake_grb)), patch.object(
            single_tree, 'handle_lazy_main_feasible_solution_gurobi'
        ), patch.object(
            single_tree, 'get_integer_solution', return_value=(1,)
        ):
            single_tree.LazyOACallback_gurobi(
                cb_model, cb_opt, fake_grb.Callback.MIPSOL, mindtpy_solver, config
            )
        self.assertEqual(cb_opt.cbLazy.call_count, 2)

    def test_lazy_gurobi_callback_new_solution_records_cut_indices(self):
        fake_grb = SimpleNamespace(
            Callback=SimpleNamespace(MIPSOL=1, MIPSOL_OBJBND=2),
            Param=SimpleNamespace(SolutionNumber='SolutionNumber'),
        )
        cb_model = make_cut_model()
        cb_opt = SimpleNamespace(
            _solver_model=SimpleNamespace(terminate=MagicMock(), SolCount=1, PoolObjVal=0),
            cbGetSolution=MagicMock(),
            cbGet=lambda key: 2.0,
            cbLazy=MagicMock(),
        )
        mindtpy_solver = SimpleNamespace(
            should_terminate=False,
            fixed_nlp=make_cut_model(),
            mip=cb_model,
            update_dual_bound=MagicMock(),
            timing={},
            primal_bound=10.0,
            dual_bound=5.0,
            rel_gap=0.5,
            log_formatter='{0}{1}{2}{3}{4}{5}{6}',
            best_solution_found=None,
            dual_bound_improved=True,
            primal_bound_improved=False,
            dual_bound_progress=[0.0],
            integer_list=[],
            integer_solution_to_cuts_index={},
            results=SolverResults(),
            mip_iter=1,
            bounds_converged=lambda: False,
            reached_time_limit=lambda: False,
            solve_subproblem=lambda: (make_cut_model(), make_results()),
        )
        mindtpy_solver.handle_nlp_subproblem_tc = MagicMock(
            side_effect=lambda fixed_nlp, fixed_nlp_result, cb: cb_model.MindtPy_utils.cuts.oa_cuts.add(
                expr=cb_model.y >= 0
            )
        )
        config = make_config(strategy='OA', add_cuts_at_incumbent=False)
        with patch.object(single_tree, 'gurobipy', SimpleNamespace(GRB=fake_grb)), patch.object(
            single_tree, 'handle_lazy_main_feasible_solution_gurobi'
        ), patch.object(
            single_tree, 'get_integer_solution', return_value=(1,)
        ):
            single_tree.LazyOACallback_gurobi(
                cb_model, cb_opt, fake_grb.Callback.MIPSOL, mindtpy_solver, config
            )
        self.assertEqual(
            mindtpy_solver.integer_solution_to_cuts_index[(1,)], [1, 1]
        )

    def test_lazy_gurobi_callback_remaining_paths(self):
        fake_grb = SimpleNamespace(
            Callback=SimpleNamespace(MIPSOL=1, MIPSOL_OBJBND=2, MIPSOL_OBJ=3),
            Param=SimpleNamespace(SolutionNumber='SolutionNumber'),
        )
        cb_model = make_cut_model()
        cb_opt = SimpleNamespace(
            _solver_model=SimpleNamespace(terminate=MagicMock(), SolCount=1, PoolObjVal=0),
            cbGetSolution=MagicMock(),
            cbGet=lambda key: 2.0,
            cbLazy=MagicMock(),
        )

        terminating_solver = SimpleNamespace(should_terminate=True)
        with patch.object(single_tree, 'gurobipy', SimpleNamespace(GRB=fake_grb)):
            single_tree.LazyOACallback_gurobi(
                cb_model, cb_opt, fake_grb.Callback.MIPSOL, terminating_solver, make_config()
            )
        cb_opt._solver_model.terminate.assert_called_once()

        bounds_solver = SimpleNamespace(
            should_terminate=False,
            fixed_nlp=make_cut_model(),
            mip=cb_model,
            update_dual_bound=MagicMock(),
            timing=SimpleNamespace(),
            primal_bound=10.0,
            dual_bound=5.0,
            rel_gap=0.5,
            log_formatter='{0}{1}{2}{3}{4}{5}{6}',
            best_solution_found=None,
            dual_bound_improved=True,
            primal_bound_improved=False,
            dual_bound_progress=[0.0],
            jacobians=ComponentMap(),
            objective_sense=minimize,
            mip_constraint_polynomial_degree={0, 1},
            integer_list=[],
            integer_solution_to_cuts_index={},
            results=SolverResults(),
            mip_iter=1,
            bounds_converged=lambda: True,
            reached_time_limit=lambda: False,
            solve_subproblem=lambda: (make_cut_model(), make_results()),
        )
        with patch.object(single_tree, 'gurobipy', SimpleNamespace(GRB=fake_grb)), patch.object(
            single_tree, 'handle_lazy_main_feasible_solution_gurobi'
        ), patch.object(single_tree, 'add_oa_cuts') as add_oa_cuts:
            single_tree.LazyOACallback_gurobi(
                cb_model,
                cb_opt,
                fake_grb.Callback.MIPSOL,
                bounds_solver,
                make_config(strategy='OA', add_cuts_at_incumbent=True),
            )
        add_oa_cuts.assert_called_once()
        self.assertGreaterEqual(cb_opt._solver_model.terminate.call_count, 2)

        skip_regularization_solver = SimpleNamespace(
            should_terminate=False,
            fixed_nlp=make_cut_model(),
            mip=cb_model,
            update_dual_bound=MagicMock(),
            timing=SimpleNamespace(),
            primal_bound=10.0,
            dual_bound=5.0,
            rel_gap=0.5,
            log_formatter='{0}{1}{2}{3}{4}{5}{6}',
            best_solution_found=make_cut_model(),
            dual_bound_improved=False,
            primal_bound_improved=False,
            dual_bound_progress=[0.0],
            jacobians=ComponentMap(),
            objective_sense=minimize,
            mip_constraint_polynomial_degree={0, 1},
            integer_list=[],
            integer_solution_to_cuts_index={},
            results=SolverResults(),
            mip_iter=1,
            bounds_converged=lambda: False,
            reached_time_limit=lambda: False,
            solve_subproblem=lambda: (make_cut_model(), make_results()),
        )
        with patch.object(single_tree, 'gurobipy', SimpleNamespace(GRB=fake_grb)), patch.object(
            single_tree, 'handle_lazy_main_feasible_solution_gurobi'
        ):
            single_tree.LazyOACallback_gurobi(
                cb_model,
                cb_opt,
                fake_grb.Callback.MIPSOL,
                skip_regularization_solver,
                make_config(strategy='OA', add_regularization='grad_lag'),
            )

        add_regularization_solver = SimpleNamespace(
            should_terminate=False,
            fixed_nlp=make_cut_model(),
            mip=cb_model,
            update_dual_bound=MagicMock(),
            timing=SimpleNamespace(),
            primal_bound=10.0,
            dual_bound=5.0,
            rel_gap=0.5,
            log_formatter='{0}{1}{2}{3}{4}{5}{6}',
            best_solution_found=make_cut_model(),
            dual_bound_improved=True,
            primal_bound_improved=False,
            dual_bound_progress=[0.0],
            jacobians=ComponentMap(),
            objective_sense=minimize,
            mip_constraint_polynomial_degree={0, 1},
            add_regularization=MagicMock(),
            integer_list=[],
            integer_solution_to_cuts_index={},
            results=SolverResults(),
            mip_iter=1,
            bounds_converged=lambda: True,
            reached_time_limit=lambda: False,
            solve_subproblem=lambda: (make_cut_model(), make_results()),
        )
        with patch.object(single_tree, 'gurobipy', SimpleNamespace(GRB=fake_grb)), patch.object(
            single_tree, 'handle_lazy_main_feasible_solution_gurobi'
        ):
            single_tree.LazyOACallback_gurobi(
                cb_model,
                cb_opt,
                fake_grb.Callback.MIPSOL,
                add_regularization_solver,
                make_config(strategy='OA', add_regularization='grad_lag'),
            )
        add_regularization_solver.add_regularization.assert_called_once()

        goa_solver = SimpleNamespace(
            should_terminate=False,
            fixed_nlp=make_cut_model(),
            mip=cb_model,
            update_dual_bound=MagicMock(),
            timing=SimpleNamespace(),
            primal_bound=10.0,
            dual_bound=5.0,
            rel_gap=0.5,
            log_formatter='{0}{1}{2}{3}{4}{5}{6}',
            best_solution_found=None,
            dual_bound_improved=True,
            primal_bound_improved=False,
            dual_bound_progress=[0.0],
            jacobians=ComponentMap(),
            objective_sense=minimize,
            mip_constraint_polynomial_degree={0, 1},
            integer_list=[(1,)],
            integer_solution_to_cuts_index={},
            results=SolverResults(),
            mip_iter=1,
            bounds_converged=lambda: False,
            reached_time_limit=lambda: False,
        )
        with patch.object(single_tree, 'gurobipy', SimpleNamespace(GRB=fake_grb)), patch.object(
            single_tree, 'handle_lazy_main_feasible_solution_gurobi'
        ), patch.object(
            single_tree, 'get_integer_solution', return_value=(1,)
        ), patch.object(
            single_tree, 'add_no_good_cuts'
        ) as add_nogood:
            single_tree.LazyOACallback_gurobi(
                cb_model,
                cb_opt,
                fake_grb.Callback.MIPSOL,
                goa_solver,
                make_config(strategy='GOA', add_no_good_cuts=True),
            )
        add_nogood.assert_called_once()

    def test_handle_lazy_main_feasible_solution_gurobi(self):
        fake_grb = SimpleNamespace(
            Callback=SimpleNamespace(MIPSOL=1, MIPSOL_OBJBND=2, MIPSOL_OBJ=3)
        )
        cb_model = make_cut_model()
        cb_opt = SimpleNamespace(
            cbGetSolution=MagicMock(),
            cbGet=lambda key: {2: 1.5, 3: 2.0}[key],
        )
        mindtpy_solver = SimpleNamespace(
            fixed_nlp=make_cut_model(),
            mip=make_cut_model(),
            update_dual_bound=MagicMock(),
            mip_iter=1,
            primal_bound=10.0,
            dual_bound=5.0,
            rel_gap=0.5,
            timing=SimpleNamespace(),
            log_formatter='{0}{1}{2}{3}{4}{5}{6}',
        )
        with patch.object(single_tree, 'gurobipy', SimpleNamespace(GRB=fake_grb)), patch.object(
            single_tree, 'copy_var_list_values'
        ) as copy_values, patch.object(
            single_tree, 'get_main_elapsed_time', return_value=1.0
        ):
            single_tree.handle_lazy_main_feasible_solution_gurobi(
                cb_model, cb_opt, mindtpy_solver, make_config()
            )
        self.assertEqual(copy_values.call_count, 2)
        mindtpy_solver.update_dual_bound.assert_called_once_with(1.5)
