import torch 
from torch import Tensor
from typing import List, Tuple, Union, Any, Optional
import numpy as np
from itertools import product, combinations
from tqdm import tqdm
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB

class ElasticOptimizer:
    """
    Optimize the accuracy loss and latency cost of a universal generator.
    """
    def __init__(self, MIP_solver = None, **solver_kwargs) -> None:
        self.MIP_solver = MIP_solver
        self.solver_kwargs = solver_kwargs

        self.optimize_objective = 'multi' # choose from ['multi', 'single']

        self.plan_limit = False

    def set_construct_num_plan_limit(self, num_bag, num_layer, num_head, limit_num):
        self.plan_limit = True

        self.num_bag = num_bag
        self.limit_num_list = [limit_num] * num_bag
        self.num_layer = num_layer
        self.num_head = num_head

    def _gp_construct_num_plan_limit_per_group(self, group_num_list: List[int], group_size_list: List[int], plan_num_list: List[int], limit_num_list: List[int], variables: gp.Var, optimization_model: gp.Model) -> List[np.ndarray]:
        """
        Add constraints to limit the number of plans for each group.
        Input:
            group_num_list: List of int, the number of groups, e.g., number of layers
            group_size_list: List of int, the size of each group, e.g., number of heads
            plan_num_list: List of int, the number of plans for each bag, e.g., number of configurations for each head
            plan_limit_list: List of int, the plan limit for each group, e.g., number of different configurations for all heads in each layer
            variables: Union[Any], the variables to construct the constraint, shape is [num_variable, 1]
            optimization_model: gp.Model, the optimization model
        """
        m = optimization_model
        
        assert sum([size * num * plan for size, num, plan in zip(group_size_list, group_num_list, plan_num_list)]) == variables.shape[0]
        
        num_bag = len(group_size_list)
        assert num_bag == len(limit_num_list)
        assert num_bag == len(group_num_list)

        # variable_num = sum(variable_num_list)
        variable_pointer = 0
        for bag_id in range(num_bag):
            group_size = group_size_list[bag_id]
            group_num = group_num_list[bag_id]
            plan_num = plan_num_list[bag_id]
            limit_num = limit_num_list[bag_id]

            variable_num = group_num*group_size*plan_num
            variable = variables[variable_pointer:variable_pointer+variable_num, :]
        
            variable = variable.reshape(group_num, group_size, plan_num)

            for group_id in range(group_num):
                select_this_plan = m.addVars(plan_num, vtype=GRB.BINARY, name=f"select_this_plan_group_{group_id}")
                for plan_id in range(plan_num):
                    m.addConstr(select_this_plan[plan_id] == gp.or_([variable[group_id, bag, plan_id] for bag in range(group_size)]), name=f"select_this_plan_group_{group_id}_plan_{plan_id}")
                m.addConstr(select_this_plan.sum() <= limit_num, name=f"limit_num_group_{group_id}")
            
            variable_pointer += variable_num


    def _construct_problem(self, accuracy_loss_list: List[Tensor], latency_cost_list: List[Tensor]) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct the optimization problem of a universal generator.
        Input:
            accuracy_loss_list: List of tensor, each tensor of shape [num_length, keep_dim, num_plan_config]
            latency_cost_list: List of tensor, each tensor of shape [num_length, keep_dim, num_plan_config]
        Output:
            num_variable: int, the number of variables in the optimization problem
            H: np.ndarray, the constraint matrix of shape [keep_dim, num_variable]
            D: np.ndarray, the latency cost matrix of shape [num_latency_length, num_variable]
            Z: np.ndarray, the accuracy loss matrix of shape [num_accuracy_length, num_variable]
        """
        assert len(accuracy_loss_list) == len(latency_cost_list)
        
        # assert the 2nd dimension of all tensors in the list are not 0
        assert all(tensor.shape[1] != 0 for tensor in accuracy_loss_list)
        assert all(tensor.shape[1] != 0 for tensor in latency_cost_list)

        H_constraint = []
        D_constraint = []
        Z_objective = []

        num_variable = 0

        for accuracy_loss, latency_cost in zip(accuracy_loss_list, latency_cost_list):
            # take the first 
            keep_dim, num_config = accuracy_loss.shape[-2:]
            assert keep_dim == latency_cost.shape[-2]
            assert num_config == latency_cost.shape[-1]
            
            # Construct a CVXPY problem.
            # x is all possible choices, which is either 0 or 1

            # Constrain, choose one config for each bag
            # H_{keep_dim, latency_cost * keep_dim} in 2 dimension. H @ x = 1
            # [[1,..,1,0,...,0,0,...,0],
            #  [0,..,0,1,...,1,0,...,0],
            #  [0,..,0,0,...,0,1,...,1]]
            H = [torch.ones(1, num_config) for _ in range(keep_dim)]
            H = concat_as_diagonal_tensor(H)

            # D_{keep_dim, latency_cost} in length dimension
            num_latency_length = latency_cost.shape[0]
            D = latency_cost.reshape(num_latency_length, keep_dim * num_config)

            # Z_{keep_dim, latency_cost} in length dimension
            num_accuracy_length = accuracy_loss.shape[0]
            Z = accuracy_loss.reshape(num_accuracy_length, keep_dim * num_config)
        
            # append this bag
            H_constraint.append(H)
            D_constraint.append(D)
            Z_objective.append(Z)
            num_variable += keep_dim * num_config
        
        H = concat_as_diagonal_tensor(H_constraint).cpu().numpy() # shape [keep_dim, num_variable] 
        D = torch.cat(D_constraint, dim=1).cpu().numpy() # shape [num_latency_length, num_variable]
        Z = torch.cat(Z_objective, dim=1).cpu().numpy() # shape [num_accuracy_length, num_variable]

        return num_variable, H, D, Z

    def _multi_objective_gurobi_solver(self, num_variable: int, H: np.ndarray, D: np.ndarray, Z: np.ndarray, latency_bound: np.ndarray, latency_lower_bound: Optional[np.ndarray]) -> np.ndarray:
        """
        Solve the multi-objective optimization problem.
        Input:
            num_variable: int, the number of variables in the optimization problem
            H: np.ndarray, the constraint matrix of shape [keep_dim, num_variable]
            D: np.ndarray, the latency cost matrix of shape [num_latency_length, num_variable]
            Z: np.ndarray, the accuracy loss matrix of shape [num_accuracy_length, num_variable]
            latency_bound: np.ndarray of float, the latency bounds for different length
            latency_lower_bound: optional np.ndarray of float, the latency lower bounds for different length
        Output:
            onehot_select: np.ndarray, the selected configuration for each bag of shape [num_variable, num_pareto_front_point]
        """
        num_points_for_each_objective = 5
        num_objective = Z.shape[0]

        # Create a new model
        # check if time_limit is a key in the dictionary
        if 'time_limit' in self.solver_kwargs:
            env = gp.Env()
            time_limit = self.solver_kwargs["time_limit"]
            print(f"Set the time limit of the optimization to {time_limit}")
            env.setParam('TimeLimit', time_limit)
            m = gp.Model("universalOptimizer", env=env)
        else:
            print("does not set time limit")
            m = gp.Model("universalOptimizer")
            pass

        m.Params.LogToConsole = 0

        # Define variables
        x = m.addMVar(shape=(num_variable, 1), vtype=GRB.BINARY, name="x")

        # Define constraints
        m.addConstr((D @ x <= latency_bound.reshape(-1,1)), name="latency")
        if latency_lower_bound is not None:
            m.addConstr((D @ x >= latency_lower_bound.reshape(-1,1)), name="latency_lower")
        m.addConstr((H @ x == 1), name="onePlanEachBag")

        if self.plan_limit:
            # Add constraint to control the number of plans for each group. E.g., the group number is the number of layers, the group size is the number of heads, the plan number is the number of configurations for each head, the limit number is the number of configurations for each layer.
            print(f"Add constraint {self.limit_num_list} to control the number of plans for each group")
            self._gp_construct_num_plan_limit_per_group(group_num_list=[self.num_layer] * self.num_bag, group_size_list=[self.num_head] * self.num_bag, plan_num_list=self.num_plan_config_list, limit_num_list=self.limit_num_list, variables=x, optimization_model=m)

        # Define objectives
        objectives = Z @ x

        # find the range of different objectives
        objective_ranges = np.zeros((num_objective, num_objective)) # each column is different objectives by optimizing the single objective, each row is the range of the objective

        for i, objective in enumerate(objectives):
            m.setObjective(objective, GRB.MINIMIZE)

            # Optimize model
            m.optimize()

            if m.status != GRB.OPTIMAL:
                status_names = {getattr(GRB, attr): attr for attr in dir(GRB) if attr.isupper() and isinstance(getattr(GRB, attr), int)}
                print(status_names)
                status = status_names.get(m.status, "UNKNOWN")
                print(f"Optimization under the objective {i} is not optimal, the status code is {m.status} : {status}")
            
            # Get the optimized configuration
            optimized_config = x.X

            # Get the range of all objectives
            objective_ranges[:, i] = np.squeeze(objectives.getValue())

        print("Optimizer: objective ranges")
        print(objective_ranges)

        num_ranges_for_each_objective = num_points_for_each_objective - 1

        objective_constraints = np.linspace(np.min(objective_ranges, axis=1), np.max(objective_ranges, axis=1), num_points_for_each_objective).T

        range_index_list = [[i for i in range(num_ranges_for_each_objective)] for _ in range(num_objective)]

        optimized_config_dict = dict()
        optimized_objective_dict = dict()

        for primary_objective_index, objective in enumerate(tqdm(objectives, total=objectives.shape[0])):
            # set the primary objective
            m.setObjective(objective, GRB.MINIMIZE)

            # iterate through all the ranges of other objectives
            constraint_objective_indices = [i for i in range(num_objective) if i != primary_objective_index]
            constraint_range_index_list = [range_index_list[i] for i in constraint_objective_indices]
            epsilon = 0.0
            for range_index_combination in product(*constraint_range_index_list):
                # add other objectives as constraints
                for i, index in enumerate(range_index_combination):
                    constraint_objective_index = constraint_objective_indices[i]
                    m.addConstr((objectives[constraint_objective_index, 0] >= objective_constraints[constraint_objective_indices[i], index] - epsilon), name=f"constraint_objective{i}_range{index}_lower")
                    m.addConstr((objectives[constraint_objective_index, 0] <= objective_constraints[constraint_objective_indices[i], index+1] + epsilon), name=f"constraint_objective{i}_range{index}_upper")

                # optimize under this objective
                m.optimize()

                if m.status == GRB.OPTIMAL:
                    # Get the optimized configuration
                    optimized_config = x.X
                    # Get the range of all objectives
                    objective_values = objectives.getValue()
                    # print(objective_values)

                    range_index_key = [i for i in range_index_combination]
                    range_index_key.insert(primary_objective_index, -1)
                    range_index_key = tuple(range_index_key)

                    optimized_config_dict[range_index_key] = optimized_config.reshape(-1)
                    optimized_objective_dict[range_index_key] = objective_values.reshape(-1)
                else:
                    status_names = {getattr(GRB, attr): attr for attr in dir(GRB) if attr.isupper() and isinstance(getattr(GRB, attr), int)}
                    print(status_names)
                    status = status_names.get(m.status, "UNKNOWN")
                    print(f"Optimization under the objective {primary_objective_index} with the range {range_index_combination} is not optimal, the status is {m.status} : {status}")
                
                # remove the constraints
                for i, index in enumerate(range_index_combination):
                    m.remove(m.getConstrByName(f"constraint_objective{i}_range{index}_lower"))
                    m.remove(m.getConstrByName(f"constraint_objective{i}_range{index}_upper"))

        # print(optimized_config_dict)
        print("optimizer: multiple objectives optimization results")
        print(optimized_objective_dict)
            
        # collect all the objective values and indices
        optimized_objective_list = []
        optimized_config_list = []
        for k in optimized_objective_dict.keys():
            optimized_objective_list.append(optimized_objective_dict[k])
            optimized_config_list.append(optimized_config_dict[k])
        optimized_objective_array = np.stack(optimized_objective_list, axis=0)
        optimized_config_array = np.stack(optimized_config_list, axis=0)

        print(f"optimizer: number of optimized objective is {optimized_objective_array.shape[0]}")

        # find the pareto front in optimized_objective_array, the objective is minimization
        # the pareto front is the points that are not dominated by any other points
        dominance_matrix = np.logical_and(
            np.all(optimized_objective_array[:, None] <= optimized_objective_array[None, :], axis=2),
            np.any(optimized_objective_array[:, None] < optimized_objective_array[None, :], axis=2)
        )
        is_pareto_point = ~np.any(dominance_matrix, axis=0)
        pareto_front_objective_array = optimized_objective_array[is_pareto_point]

        print(f"optimizer: number of pareto front points is {pareto_front_objective_array.shape[0]}")
        print("optimizer: pareto_front_objective_array")
        print(pareto_front_objective_array)

        pareto_front_config_array = (optimized_config_array[is_pareto_point]).T

        return pareto_front_config_array  

    def _single_objective_cvxpy_solver(self, num_variable: int, H: np.ndarray, D: np.ndarray, Z: np.ndarray, latency_bound: np.ndarray, latency_lower_bound: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve the single objective optimization problem.
        Input:
            num_variable: int, the number of variables in the optimization problem
            H: np.ndarray, the constraint matrix of shape [keep_dim, num_variable]
            D: np.ndarray, the latency cost matrix of shape [num_latency_length, num_variable]
            Z: np.ndarray, the accuracy loss matrix of shape [num_accuracy_length, num_variable]
            latency_bound: np.ndarray of float, the latency bounds for different length
            latency_lower_bound: optional np.ndarray of float, the latency lower bounds for different length
        Output:
            onehot_select: np.ndarray, the selected configuration for each bag of shape [num_variable, 1]
        """
        # Variable
        x = cp.Variable((num_variable, 1), boolean=True) 
        # Objective. 
        objective = cp.Minimize(cp.sum(Z @ x))
        # x_{keep_dim, latency_cost} in 1 dimension
        constraints = [D @ x <= latency_bound.reshape(-1,1), H @ x == 1]
        if latency_lower_bound is not None:
            constraints.append(D @ x >= latency_lower_bound.reshape(-1,1))

        # Define and solve the CVXPY problem. (MIP_solver=cp.ECOS_BB)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.MIP_solver, **self.solver_kwargs)

        print("Status: ", prob.status)
        # print("The optimal value is", prob.value)
        # print("A solution x is")
        # print(np.round(x.value))

        # Select the best config for each bag
        onehot_select = np.round(x.value) # shape [num_variable, 1]

        return onehot_select

    def optimize(self, accuracy_loss_list: List[Tensor], latency_cost_list: List[Tensor], latency_bound: Tensor, latency_lower_bound: Optional[Tensor] = None) -> Tensor:
        """
        Optimize the accuracy loss and latency cost of a universal generator. The list indicates plans from different generators.
        Input:
            accuracy_loss_list: List of tensor, each tensor of shape [num_length, keep_dim, num_plan_config]
            latency_cost_list: List of tensor, each tensor of shape [num_length, keep_dim, num_plan_config]
            latency_bound: Tensor of float, the latency bounds for different length
        """
        assert len(accuracy_loss_list) == len(latency_cost_list)

        # get the size info
        num_accuracy_length = accuracy_loss_list[0].shape[0] # all bags share the same number of accuracy length
        assert all(tensor.shape[0] == num_accuracy_length for tensor in accuracy_loss_list)
        num_latency_length = latency_cost_list[0].shape[0] # all bags share the same number of latency length
        assert all(tensor.shape[0] == num_latency_length for tensor in latency_cost_list)
        keep_dim = accuracy_loss_list[0].shape[1] # all bags share the same keep_dim
        assert all(tensor.shape[1] == keep_dim for tensor in accuracy_loss_list)
        assert all(tensor.shape[1] == keep_dim for tensor in latency_cost_list)
        num_plan_config_list = [tensor.shape[2] for tensor in accuracy_loss_list] # the num plan config from accuracy loss and latency cost should be the same
        assert all(accuracy_loss.shape[2]==latency_cost.shape[2] for accuracy_loss, latency_cost in zip(accuracy_loss_list, latency_cost_list))

        if self.plan_limit:
            self.num_plan_config_list = num_plan_config_list


        # construct the optimization problem
        num_variable, H, D, Z  = self._construct_problem(accuracy_loss_list, latency_cost_list)

        # Solve the overall optimization problem
        print("Solving the overall optimization problem of scale: {} variables".format(num_variable))
        if self.optimize_objective == 'single':
            onehot_select = self._single_objective_cvxpy_solver(num_variable, H, D, Z, latency_bound, latency_lower_bound) # shape [num_variable, 1]
        elif self.optimize_objective == 'multi':
            onehot_select = self._multi_objective_gurobi_solver(num_variable, H, D, Z, latency_bound.numpy(), latency_lower_bound.numpy() if latency_lower_bound is not None else None) # shape [num_variable, num_pareto_front_point]
        else:
            raise ValueError(f"optimize_objective should be one of ['single', 'multi'], but got {self.optimize_objective}")

        num_plan = onehot_select.shape[1]
        
        # convert to list index
        currect_variable = 0
        opt_cofig_index_list = []
        for i in range(len(accuracy_loss_list)):
            keep_dim, num_config = accuracy_loss_list[i].shape[-2:]
            current_onehot_select = onehot_select[currect_variable:currect_variable+keep_dim*num_config, :].reshape(keep_dim, num_config, num_plan).transpose(2, 0, 1) # shape [num_plan, keep_dim, num_config]
            currect_variable += keep_dim * num_config
        
            # select the config_id with onehot_select
            opt_config_id = (torch.from_numpy(current_onehot_select).to(int) @ torch.arange(num_config).to(int).reshape(-1,1)).reshape(num_plan, -1)
            opt_cofig_index_list.append(opt_config_id)

        return opt_cofig_index_list

class UniversalOptimizer:
    """
    Optimize the accuracy loss and latency cost of a universal generator.
    """
    def __init__(self, MIP_solver = None, **kwargs) -> None:
        self.MIP_solver = MIP_solver
        self.kwargs = kwargs

    def optimize(self, accuracy_loss_list: List[Tensor], latency_cost_list: List[Tensor], latency_bound: float, latency_lower_bound=None) -> Tensor:
        """
        Optimize the accuracy loss and latency cost of a universal generator.
        Input:
            accuracy_loss_list: List of tensor, each tensor of shape [keep_dim, num_plan_config]
            latency_cost_list: List of tensor, each tensor of shape [keep_dim, num_plan_config]
            latency_bound: float, the latency bound
        """

        assert len(accuracy_loss_list) == len(latency_cost_list)
        
        # assert the 2nd dimension of all tensors in the list are not 0
        assert all(tensor.shape[1] != 0 for tensor in accuracy_loss_list)
        assert all(tensor.shape[1] != 0 for tensor in latency_cost_list)
        assert latency_lower_bound is None or latency_lower_bound <= latency_bound

        H_constraint = []
        D_constraint = []
        Z_objective = []

        num_variable = 0

        for accuracy_loss, latency_cost in zip(accuracy_loss_list, latency_cost_list):
            
            keep_dim, num_config = accuracy_loss.shape
            assert keep_dim == latency_cost.shape[0]
            assert num_config == latency_cost.shape[1]
            
            # Construct a CVXPY problem.
            # x is all possible choices, which is either 0 or 1

            # Constrain, choose one config for each bag
            # H_{keep_dim, latency_cost * keep_dim} in 2 dimension. H @ x = 1
            # [[1,..,1,0,...,0,0,...,0],
            #  [0,..,0,1,...,1,0,...,0],
            #  [0,..,0,0,...,0,1,...,1]]
            H = [torch.ones(1, num_config) for _ in range(keep_dim)]
            H = concat_as_diagonal_tensor(H)

            # D_{keep_dim, latency_cost} in 1 dimension
            D = latency_cost.reshape(1, keep_dim * num_config)

            # Z_{keep_dim, latency_cost} in 1 dimension
            Z = accuracy_loss.reshape(1, keep_dim * num_config)
        
            # append this bag
            H_constraint.append(H)
            D_constraint.append(D)
            Z_objective.append(Z)
            num_variable += keep_dim * num_config
        
        H = concat_as_diagonal_tensor(H_constraint).cpu().numpy()
        D = torch.cat(D_constraint, dim=1).cpu().numpy()
        Z = torch.cat(Z_objective, dim=1).cpu().numpy()

        # Solve the overall optimization problem
        print("Solving the overall optimization problem of scale: {} variables".format(num_variable))
        
        # Variable
        x = cp.Variable((num_variable, 1), boolean=True) 
        # Objective. 
        objective = cp.Minimize(Z @ x)
        # x_{keep_dim, latency_cost} in 1 dimension
        constraints = [D @ x <= latency_bound, H @ x == 1]
        if latency_lower_bound is not None:
            constraints.append(D @ x >= latency_lower_bound)
        
        # Define and solve the CVXPY problem. (MIP_solver=cp.ECOS_BB)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.MIP_solver, **self.kwargs)

        print("Status: ", prob.status)
        # print("The optimal value is", prob.value)
        # print("A solution x is")
        # print(np.round(x.value))

        # Select the best config for each bag
        onehot_select = np.round(x.value)
        
        currect_variable = 0
        opt_cofig_index_list = []
        for i in range(len(accuracy_loss_list)):
            keep_dim, num_config = accuracy_loss_list[i].shape
            current_onehot_select = onehot_select[currect_variable:currect_variable+keep_dim*num_config].reshape(keep_dim, num_config)
            currect_variable += keep_dim * num_config
        
            # select the config_id with onehot_select
            opt_config_id = (torch.from_numpy(current_onehot_select).to(int) @ torch.arange(num_config).to(int).reshape(-1,1)).reshape(-1)
            opt_cofig_index_list.append(opt_config_id)

        return opt_cofig_index_list

def concat_as_diagonal_tensor(tensor_list: List[Tensor]) -> Tensor:
    """
    Concat a list of tensors as a larger tensor. The tensors are put on the diagonal of the larger tensor.
    Input:
        tensor_list: list of tensor of different shapes [N_i, M_i]
    Output:
        large tensor of shape [\sum_i N_i, \sum_i M_i]
    Example:
        tensor_list = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8], [9, 10]])]
        concat_as_diagonal_tensor(tensor_list)
        >>> tensor([[ 1,  2,  0,  0],
                    [ 3,  4,  0,  0],
                    [ 0,  0,  5,  6],
                    [ 0,  0,  7,  8],
                    [ 0,  0,  9, 10]])
    """
    assert all(len(tensor.shape)==2 for tensor in tensor_list), "all tensors in tensor_list should be 2D"

    output_tensor = torch.zeros(sum([tensor.shape[0] for tensor in tensor_list]), sum([tensor.shape[1] for tensor in tensor_list]))
    row_start = 0
    col_start = 0
    for tensor in tensor_list:
        output_tensor[row_start:row_start+tensor.shape[0], col_start:col_start+tensor.shape[1]] = tensor
        row_start += tensor.shape[0]
        col_start += tensor.shape[1]
    return output_tensor