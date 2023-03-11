from django.shortcuts import render
from django.http import JsonResponse,HttpResponse
import numpy as np
import math

max_units=20
lembda=13
k=15
rev_per_unit=15
var_cost=3
hold_cost = 1





def home(request):
    context={}
    return render(request,'base.html',context)

def simulation_parameter(request):
    if request.method=='POST':
        max_units=int(request.POST['max_units'])
        lembda=float(request.POST['poisson_paramter'])
        k=float(request.POST['fixed_cost'])
        rev_per_unit=float(request.POST['revenue_per_unit'])
        var_cost=float(request.POST['var_cost_per_cost'])
        hold_cost=float(request.POST['hold_cost'])
        # max_units=20
        # lembda=5
        # k=15
        # rev_per_unit=15
        # var_cost=3
        # print('max_units')
        # hold_cost = 1
        def poisson(k):
            if k>max_units:
                return 0
            prob = (np.power(lembda,k)*np.exp(-lembda))/math.factorial(k)
            return prob

        def prob_demand_exceeds(u):
            if u == 0:
                return 1

            tot_sum = 0
            cur_u = u
            while 1:
                cur_sum = 0
                cur_sum += tot_sum
                tot_sum += poisson(cur_u)
                cur_u += 1
                if abs(cur_sum - tot_sum) < 1e-9:
                    break
            return tot_sum

        def order_cost(u):
            cost = 0
            if u!=0:
                cost = k+var_cost*u
            return cost

        def holding_cost(u):
            return u*hold_cost

        def revenue(u):
            return rev_per_unit*u

        def expected_revenue(u):
            ans = 0
            for j in range(u):
                ans += revenue(j)*poisson(j)
            ans += revenue(u)*prob_demand_exceeds(u)
            return ans

        def reward(s,a):
            r = expected_revenue(s+a) - order_cost(a) - holding_cost(s)
            return r

        def transition_prob(j,s,a):
            if j>s+a:
                return 0
            if j<=s+a and j>0:
                return poisson(s+a-j)
            if j==0:
                return prob_demand_exceeds(s+a)

        def make_transition_matrix():
            transition_matrix = {}
            for i in range(max_units+1):
                d = {}
                for j in range(max_units+1):
                    l = []
                    if i+j > max_units:
                        continue
                    for k in range(max_units+1):
                        p = transition_prob(k,i,j)
                        l.append(p)
                    d[j] = l
                transition_matrix[i] = d
            return transition_matrix

        def make_reward_matrix():
            reward_matrix = {}
            for i in range(max_units+1):
                d = []
                for j in range(max_units+1):
                    if i+j > max_units:
                        continue
                    d.append(reward(i,j))
                reward_matrix[i] = d
            return reward_matrix

        def make_P_dict():
            transition_matrix = make_transition_matrix()
            reward_matrix = make_reward_matrix()
            p = {}
            for i in range(max_units+1):
                d = {}

                for j in range(len(transition_matrix[i])):
                    l = []

                    for k in range(len(transition_matrix[i][j])):
                        x = []
                        x.append(transition_matrix[i][j][k])
                        x.append(k)
                        x.append(reward_matrix[i][j])
                        x.append(False)
                        x = tuple(x)
                        l.append(x)
                    d[j] = l
                p[i] = d
            return p

        def val_iter(gamma,max_iter=1000,tolerence = 1e-6):
            P_env = make_P_dict()
            n_states = len(P_env)
            values = np.zeros(n_states)
            policy = np.zeros(n_states)
            iter = 0
            gap_vals = []
            max_val_state = []
            while 1:
                old_values = np.array([values[i] for i in range(n_states)])
                iter += 1
                for i in range(n_states):       
                    actions_i = P_env[i]           
                    max_val = 0
                    best_act = 0

                    for j in range(len(actions_i)):     
                        action_i_j = actions_i[j]       

                        val_j_action = 0
                        for k in range(len(action_i_j)):
                            action_i_j_k = action_i_j[k]    
                            p = action_i_j_k[0]
                            s_prime = action_i_j_k[1]
                            reward = action_i_j_k[2]
                            val_j_action += p*(reward+gamma*values[s_prime])

                        if max_val < val_j_action:
                            max_val = val_j_action
                            best_act = j

                    values[i] = max_val
                    policy[i] = best_act

                max_val_state.append(max(values))
                gap_vals.append(max(abs(old_values-values)))
                if max_iter < iter or max(abs(old_values-values)) < tolerence:
                    break

            return list(values),list(policy)
        
        
        optimal_values,optimal_policy = val_iter(0.99)
        context={}
        return JsonResponse({"data":{'ov':optimal_values,'op':optimal_policy}})
    else:
        return render(request,'home.html')

def inv_system(request):
    context={}
    return render(request,'inv_manage.html',context)

