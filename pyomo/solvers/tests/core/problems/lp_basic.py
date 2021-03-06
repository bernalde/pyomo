#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# A LP model adapted from the Wikipedia description of the simplex algorithm:
#  http://en.wikipedia.org/wiki/Simplex_algorithm
#

from pyomo.core import ConcreteModel, Var, Objective, Constraint, NonNegativeReals

model = ConcreteModel()

model.x = Var(within=NonNegativeReals)
model.y = Var(within=NonNegativeReals)
model.z = Var(within=NonNegativeReals)

model.Z = Objective(expr=-2*model.x -3*model.y -4*model.z)

model.c1 = Constraint(expr=3*model.x + 2*model.y +    model.z <= 10)
model.c2 = Constraint(expr=2*model.x + 5*model.y + 3* model.z <= 15)
