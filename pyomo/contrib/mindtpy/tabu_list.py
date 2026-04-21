# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
#

from pyomo.common.dependencies import attempt_import, UnavailableClass

cplex, cplex_available = attempt_import('cplex')


def _get_callback_integer_solution(discrete_variable_list, solver_var_map, get_values):
    """Extract the incumbent integer solution from a callback context."""
    return tuple(
        int(round(get_values(solver_var_map[var]))) for var in discrete_variable_list
    )


def _should_reject_incumbent(single_tree, curr_int_sol, integer_list):
    """Return whether the incumbent should be rejected."""
    return single_tree or curr_int_sol in set(integer_list)


class IncumbentCallback_cplex(
    cplex.callbacks.IncumbentCallback if cplex_available else UnavailableClass(cplex)
):
    """Inherent class in Cplex to call Incumbent callback."""

    def __call__(self):
        """
        This is an inherent function in LazyConstraintCallback in CPLEX.
        This callback will be used after each new potential incumbent is found.
        https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/ilog.odms.cplex.help/refpythoncplex/html/cplex.callbacks.IncumbentCallback-class.html
        IncumbentCallback will be activated after Lazyconstraint callback, when the potential incumbent solution is satisfies the lazyconstraints.
        TODO: need to handle GOA same integer combination check in lazyconstraint callback in single_tree.py
        """
        mindtpy_solver = self.mindtpy_solver
        opt = self.opt
        config = self.config
        mindtpy_solver.curr_int_sol = _get_callback_integer_solution(
            mindtpy_solver.mip.MindtPy_utils.discrete_variable_list,
            opt._pyomo_var_to_solver_var_map,
            self.get_values,
        )
        if _should_reject_incumbent(
            config.single_tree, mindtpy_solver.curr_int_sol, mindtpy_solver.integer_list
        ):
            self.reject()
