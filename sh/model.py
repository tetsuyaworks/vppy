import warnings
from collections import OrderedDict
import datetime
import random
import multiprocessing
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SingleHormoneBloodGlucoseModel:

    param_set = OrderedDict()
    # Fc01: Non-insulin mediated glucose uptake above 4.5 mmol/L ([mmol/kg]/min)
    param_set["Fc01"] = OrderedDict()
    param_set["Fc01"]["default"] = 0.0097
    param_set["Fc01"]["min"] = 0.0001
    param_set["Fc01"]["max"] = 0.02
    param_set["Fc01"]["type"] = float

    # Vdg: Volume of distribution of glucose (L/kg)
    param_set["VdG"] = OrderedDict()
    param_set["VdG"]["default"] = 0.16
    param_set["VdG"]["min"] = 0.01
    param_set["VdG"]["max"] = 0.3
    param_set["VdG"]["type"] = float

    # k12: Rate constant for glucose transfer from Q2 to Q1 (min^-1)
    param_set["k12"] = OrderedDict()
    param_set["k12"]["default"] = 0.066
    param_set["k12"]["min"] = 0.001
    param_set["k12"]["max"] = 0.1
    param_set["k12"]["type"] = float

    # Ag: Carb bioavailability (unitless)
    param_set["Ag"] = OrderedDict()
    param_set["Ag"]["default"] = 0.8
    param_set["Ag"]["min"] = 0.1
    param_set["Ag"]["max"] = 2.0
    param_set["Ag"]["type"] = float

    # tmaxG: time-to-maximum of carb absorption (min)
    param_set["tmaxG"] = OrderedDict()
    param_set["tmaxG"]["default"] = 40
    param_set["tmaxG"]["min"] = 1
    param_set["tmaxG"]["max"] = 80
    param_set["tmaxG"]["type"] = int

    # EGP0: Endogenous glucose production maximum ([mmol/kg]/min)
    param_set["EGP0"] = OrderedDict()
    param_set["EGP0"]["default"] = 0.0161
    param_set["EGP0"]["min"] = 0.0001
    param_set["EGP0"]["max"] = 0.03
    param_set["EGP0"]["type"] = float

    # tmaxI: time-to-maximum of rapid-acting insulin absorption
    param_set["tmaxI"] = OrderedDict()
    param_set["tmaxI"]["default"] = 55
    param_set["tmaxI"]["min"] = 1
    param_set["tmaxI"]["max"] = 100
    param_set["tmaxI"]["type"] = int

    # Ke: Elimination rate of insulin (min^-1)
    param_set["Ke"] = OrderedDict()
    param_set["Ke"]["default"] = 0.138
    param_set["Ke"]["min"] = 0.001
    param_set["Ke"]["max"] = 0.3
    param_set["Ke"]["type"] = float

    # VdI: Volume of distribution of insulin (L/kg)
    param_set["VdI"] = OrderedDict()
    param_set["VdI"]["default"] = 0.12
    param_set["VdI"]["min"] = 0.01
    param_set["VdI"]["max"] = 0.3
    param_set["VdI"]["type"] = float

    # ka1: Rate constant for elimination of insulin effect from x1 (min^-1)
    param_set["ka1"] = OrderedDict()
    param_set["ka1"]["default"] = 0.006
    param_set["ka1"]["min"] = 0.001
    param_set["ka1"]["max"] = 0.01
    param_set["ka1"]["type"] = float

    # ka2: Rate constant for elimination of insulin effect from x2 (min^-1)
    param_set["ka2"] = OrderedDict()
    param_set["ka2"]["default"] = 0.06
    param_set["ka2"]["min"] = 0.01
    param_set["ka2"]["max"] = 0.1
    param_set["ka2"]["type"] = float

    # ka3: Rate constant for elimination of insulin effect from x3 (min^-1)
    param_set["ka3"] = OrderedDict()
    param_set["ka3"]["default"] = 0.03
    param_set["ka3"]["min"] = 0.01
    param_set["ka3"]["max"] = 0.1
    param_set["ka3"]["type"] = float

    # Sf1: Sensitivity factor for glucose distribution (x1) ([mU.L.min]^-2)
    param_set["Sf1"] = OrderedDict()
    param_set["Sf1"]["default"] = 0.00542
    param_set["Sf1"]["min"] = 0.00001
    param_set["Sf1"]["max"] = 0.01
    param_set["Sf1"]["type"] = float

    # Sf2: Sensitivity factor for insulin mediated glucose utilization (x2) ([mU.L.min]^-2)
    param_set["Sf2"] = OrderedDict()
    param_set["Sf2"]["default"] = 0.00082
    param_set["Sf2"]["min"] = 0.00001
    param_set["Sf2"]["max"] = 0.001
    param_set["Sf2"]["type"] = float

    # Sf3: Sensitivity factor for suppression of endogenous glucose production (x3) ([mU.L.min]^-1)
    param_set["Sf3"] = OrderedDict()
    param_set["Sf3"]["default"] = 0.052
    param_set["Sf3"]["min"] = 0.001
    param_set["Sf3"]["max"] = 0.1
    param_set["Sf3"]["type"] = float

    # Ratio between the TDIR and TDIR_basal.
    # A ratio of 2 indicates basal is 50% of TDIR (1/0.5). Ratio of 1.78 indicates
    # that basal insulin comprises 56.18% of total daily insulin (1/0.5618)
    param_set["TDIR_basal_rate"] = OrderedDict()
    param_set["TDIR_basal_rate"]["default"] = 1.78
    param_set["TDIR_basal_rate"]["min"] = 0.01
    param_set["TDIR_basal_rate"]["max"] = 4.0
    param_set["TDIR_basal_rate"]["type"] = float

    # percentage of pre-meal bolus  [unitless: 0-1]
    param_set["Ip"] = OrderedDict()
    param_set["Ip"]["default"] = 1.0
    param_set["Ip"]["min"] = 0.001
    param_set["Ip"]["max"] = 1.0
    param_set["Ip"]["type"] = float

    # time-to-maximum rescure carb absorption [min]
    param_set["tmax_resc"] = OrderedDict()
    param_set["tmax_resc"]["default"] = 20
    param_set["tmax_resc"]["min"] = 1
    param_set["tmax_resc"]["max"] = 40
    param_set["tmax_resc"]["type"] = int

    # Rescue carbs given for glucose < 70 mg/dL
    param_set["Thr_resc"] = OrderedDict()
    param_set["Thr_resc"]["default"] = 70
    param_set["Thr_resc"]["min"] = 1
    param_set["Thr_resc"]["max"] = 70
    param_set["Thr_resc"]["type"] = int

    # 20 g of carbs given when glucose <  70 mg/dL
    param_set["Carbs_resc"] = OrderedDict()
    param_set["Carbs_resc"]["default"] = 20
    param_set["Carbs_resc"]["min"] = 1
    param_set["Carbs_resc"]["max"] = 70
    param_set["Carbs_resc"]["type"] = int

    # Window for lower insulin dosed is 40 minutes after hypo
    param_set["Win_resc"] = OrderedDict()
    param_set["Win_resc"]["default"] = 40
    param_set["Win_resc"]["min"] = 1
    param_set["Win_resc"]["max"] = 80
    param_set["Win_resc"]["type"] = int

    # Insulin is reduced to 25%
    param_set["IIR_red_resc"] = OrderedDict()
    param_set["IIR_red_resc"]["default"] = 0.25
    param_set["IIR_red_resc"]["min"] = 0.01
    param_set["IIR_red_resc"]["max"] = 0.5
    param_set["IIR_red_resc"]["type"] = float

    # for 40 minutes after a hypo
    param_set["timer_resc"] = OrderedDict()
    param_set["timer_resc"]["default"] = 40
    param_set["timer_resc"]["min"] = 1
    param_set["timer_resc"]["max"] = 80
    param_set["timer_resc"]["type"] = int

    # Rescue carb is given 20 minutes after hypo occurs
    param_set["delay_rescue_val"] = OrderedDict()
    param_set["delay_rescue_val"]["default"] = 20
    param_set["delay_rescue_val"]["min"] = 1
    param_set["delay_rescue_val"]["max"] = 40
    param_set["delay_rescue_val"]["type"] = int

    def __init__(self, params=None):
        if params:
            self.params = params
        else:
            self.params = OrderedDict()
            self.params["Fc01"] = 0.0097  # Fc01: Non-insulin mediated glucose uptake above 4.5 mmol/L ([mmol/kg]/min)
            self.params["VdG"] = 0.16  # Vdg: Volume of distribution of glucose (L/kg)
            self.params["k12"] = 0.066  # k12: Rate constant for glucose transfer from Q2 to Q1 (min^-1)
            self.params["Ag"] = 0.8  # Ag: Carb bioavailability (unitless)
            self.params["tmaxG"] = 40  # tmaxG: time-to-maximum of carb absorption (min)
            self.params["EGP0"] = 0.0161  # EGP0: Endogenous glucose production maximum ([mmol/kg]/min)
            self.params["tmaxI"] = 55  # tmaxI: time-to-maximum of rapid-acting insulin absorption
            self.params["Ke"] = 0.138  # Ke: Elimination rate of insulin (min^-1)
            self.params["VdI"] = 0.12  # VdI: Volume of distribution of insulin (L/kg)
            self.params["ka1"] = 0.006  # ka1: Rate constant for elimination of insulin effect from x1 (min^-1)
            self.params["ka2"] = 0.06  # ka2: Rate constant for elimination of insulin effect from x2 (min^-1)
            self.params["ka3"] = 0.03  # ka3: Rate constant for elimination of insulin effect from x3 (min^-1)
            # Sf1: Sensitivity factor for glucose distribution (x1) ([mU.L.min]^-2)
            self.params["Sf1"] = 0.00542
            # Sf2: Sensitivity factor for insulin mediated glucose utilization (x2) ([mU.L.min]^-2)
            self.params["Sf2"] = 0.00082
            # Sf3: Sensitivity factor for suppression of endogenous glucose production (x3) ([mU.L.min]^-1)
            self.params["Sf3"] = 0.052

            # Ratio between the TDIR and TDIR_basal.
            # A ratio of 2 indicates basal is 50% of TDIR (1/0.5). Ratio of 1.78 indicates
            # that basal insulin comprises 56.18% of total daily insulin (1/0.5618)
            self.params["TDIR_basal_rate"] = 1.78

            self.params["Ip"] = 1  # percentage of pre-meal bolus  [unitless: 0-1]
            self.params["tmax_resc"] = 20  # time-to-maximum rescure carb absorption [min]
            self.params["Thr_resc"] = 70  # Rescue carbs given for glucose < 70 mg/dL
            self.params["Carbs_resc"] = 20  # 20 g of carbs given when glucose <  70 mg/dL
            self.params["Win_resc"] = 40  # Window for lower insulin dosed is 40 minutes after hypo
            self.params["IIR_red_resc"] = 0.25  # Insulin is reduced to 25%
            self.params["timer_resc"] = 40  # for 40 minutes after a hypo
            self.params["delay_rescue_val"] = 20  # Rescue carb is given 20 minutes after hypo occurs

    def predict(self, simulation_days, weight, starting_glucose, meal_scenario, _params=None):

        def solve_ss(x, q, params):
            return np.array([
                -x[5] * q - params[0] + params[2] * x[1] + params[5] * (1 - x[7]),
                x[5] * q - params[2] * x[1] - x[6] * x[1],
                -params[9] * x[5] + (params[12] * params[9]) * x[4],
                -params[10] * x[6] + (params[13] * params[10]) * x[4],
                -params[11] * x[7] + (params[14] * params[11]) * x[4],
                x[0] - x[2] / params[6],
                x[2] / params[6] - x[3] / params[6],
                x[3] / (params[6] * params[8]) - params[7] * x[4]
            ])

        def set_initial_conditions(glucose_set_point, fc01, vdg, k12, ag, tmaxg, egp0, tmaxi, ke, vdi, ka1, ka2, ka3,
                                    sf1,
                                    sf2,
                                    sf3):
            set_point_init = glucose_set_point / 18 * 0.16
            xx0 = 0 * np.ones(8)
            fun = lambda x: solve_ss(x, set_point_init,
                                     [fc01, vdg, k12, ag, tmaxg, egp0, tmaxi, ke, vdi, ka1, ka2, ka3, sf1, sf2, sf3])
            return fsolve(fun, xx0)

        def calc_tdir(
                weight,
                fc01, vdg, k12, ag, tmaxg, egp0, tmaxi, ke, vdi, ka1, ka2, ka3, sf1, sf2, sf3,
                tdir_basal_rate, glucose_set_point
        ):
            xm_plant = set_initial_conditions(glucose_set_point, fc01, vdg, k12, ag, tmaxg, egp0, tmaxi, ke, vdi, ka1,
                                              ka2, ka3,
                                              sf1, sf2, sf3)
            iir_ss = xm_plant[0] * weight * 60 / 1000
            return 24 * tdir_basal_rate * iir_ss

        def sh_glucoregulatory(x,
                               fc01, vdg, k12, ag, tmaxg, egp0, tmaxi, ke, vdi, ka1, ka2, ka3, sf1, sf2, sf3,
                               time_space, m_e_piu, m_e_pgu, m_e_hgp):
            Sf1 = m_e_piu * m_e_pgu * sf1
            Sf2 = m_e_piu * m_e_pgu * sf2
            Sf3 = m_e_hgp * sf3

            kb1 = Sf1 * ka1  # L / mU
            kb2 = Sf2 * ka2  # L / mU
            kb3 = Sf3 * ka3  # L / mU - min^-1

            q1 = x[0]  # mmol/L
            q2 = x[1]
            s1 = x[2]
            s2 = x[3]
            i = x[4]
            x1 = x[5]
            x2 = x[6]
            x3 = x[7]

            g = q1 / vdg  # (mmol/kg)/(L/kg) == mmol/L
            # for mg/dL, multiply mmol/L by 18

            ap = np.zeros((8, 8))
            bp = np.zeros((8, 1))
            dp = np.zeros((8, 1))
            cp = np.zeros((1, 8))

            if g < 4.5:
                kc01 = fc01 / (4.5 * vdg)
                kr = 0
                kd = 0
            elif 4.5 <= g < 9:
                kc01 = 0
                kr = 0
                kd = fc01
            else:
                kc01 = 0
                kr = 0.003
                kd = fc01 - 0.003 * 9 * vdg

            ap[0, 0] = -x1 - kc01 - kr
            ap[0, 1] = k12
            ap[0, 5] = -q1
            ap[0, 7] = -egp0
            ap[1, 0] = x1
            ap[1, 1] = -(k12 + x2)
            ap[1, 5] = q1
            ap[1, 6] = -q2

            ap[2, 2] = -1 / tmaxi
            ap[3, 2] = 1 / tmaxi
            ap[3, 3] = -1 / tmaxi
            ap[4, 3] = 1 / (tmaxi * vdi)
            ap[4, 4] = -ke

            ap[5, 5] = -ka1
            ap[6, 6] = -ka2
            ap[7, 7] = -ka3
            ap[5, 4] = kb1
            ap[6, 4] = kb2
            ap[7, 4] = kb3

            bp[2, 0] = 1

            dp[0, 0] = x1 * q1 - kd + egp0
            dp[1, 0] = -x1 * q1 + x2 * q2

            cp[0, 0] = 1
            cd = cp
            dd = dp * time_space

            ad = ap * time_space + np.eye(8)
            bd = bp * time_space

            return ad, bd, cd, dd

        time_space = 5  # time_spaceを5にしないとなぜか機能しないため、5に固定する
        sim_time = int(simulation_days * 1440 / time_space)

        meal_vector = np.zeros(sim_time + 1)
        for v in meal_scenario:
            if v[0] > sim_time * time_space:
                continue
            meal_vector[int(round(v[0] / time_space))] = v[1]
        meal_time = []
        meal_amount = []

        if _params:
            fc01 = _params["Fc01"]
            vdg = _params["VdG"]
            k12 = _params["k12"]
            ag = _params["Ag"]
            tmaxg = _params["tmaxG"]
            egp0 = _params["EGP0"]
            tmaxi = _params["tmaxI"]
            ke = _params["Ke"]
            vdi = _params["VdI"]
            ka1 = _params["ka1"]
            ka2 = _params["ka2"]
            ka3 = _params["ka3"]
            sf1 = _params["Sf1"]
            sf2 = _params["Sf2"]
            sf3 = _params["Sf3"]
            tdir_basal_rate = _params["TDIR_basal_rate"]
            ip = _params["Ip"]
            tmax_resc = _params["tmax_resc"]
            thr_resc = _params["Thr_resc"]
            carbs_resc = _params["Carbs_resc"]
            win_resc = _params["Win_resc"] / time_space
            iir_red_resc = _params["IIR_red_resc"]
            timer_resc = _params["timer_resc"] / time_space
            delay_rescue_val = _params["delay_rescue_val"]
        else:
            fc01 = self.params["Fc01"]
            vdg = self.params["VdG"]
            k12 = self.params["k12"]
            ag = self.params["Ag"]
            tmaxg = self.params["tmaxG"]
            egp0 = self.params["EGP0"]
            tmaxi = self.params["tmaxI"]
            ke = self.params["Ke"]
            vdi = self.params["VdI"]
            ka1 = self.params["ka1"]
            ka2 = self.params["ka2"]
            ka3 = self.params["ka3"]
            sf1 = self.params["Sf1"]
            sf2 = self.params["Sf2"]
            sf3 = self.params["Sf3"]
            tdir_basal_rate = self.params["TDIR_basal_rate"]
            ip = self.params["Ip"]
            tmax_resc = self.params["tmax_resc"]
            thr_resc = self.params["Thr_resc"]
            carbs_resc = self.params["Carbs_resc"]
            win_resc = self.params["Win_resc"] / time_space
            iir_red_resc = self.params["IIR_red_resc"]
            timer_resc = self.params["timer_resc"] / time_space
            delay_rescue_val = self.params["delay_rescue_val"]

        glucose_set_point = 115
        tdir = calc_tdir(
            weight,
            fc01, vdg, k12, ag, tmaxg, egp0, tmaxi, ke, vdi, ka1, ka2, ka3, sf1, sf2, sf3,
            tdir_basal_rate, glucose_set_point
        )
        icr = (1700 / tdir / 3)
        bolus = ip * (meal_vector / icr) * 1000 / (weight * time_space)
        cgm_start = starting_glucose
        xm_plant = set_initial_conditions(glucose_set_point, fc01, vdg, k12, ag, tmaxg, egp0, tmaxi, ke, vdi, ka1, ka2,
                                          ka3,
                                          sf1, sf2, sf3)
        q1 = cgm_start / 18 * vdg
        xm_plant[0] = q1
        xm_plant = np.array([xm_plant]).T

        pgua_1_act = 0

        cntr_resc = win_resc
        resc_trig_cntr = 0
        time_resc = []
        ur_plant = 0
        ur_mdl = 0
        dr_plnt = 0
        dr_mdl = 0
        ins_adj_resc = np.ones(sim_time + 1)

        u_basal = (tdir / tdir_basal_rate / 24) * 1000 / weight / 60
        bg_output = []
        ins_input = []
        u_total = u_basal
        for idx in range(sim_time):
            u_total = u_basal
            if idx > 0:
                u_total = ins_adj_resc[idx] * u_basal + bolus[idx]

            # meal response
            if meal_vector[idx] > 0:
                meal_time.append(idx)
                meal_amount.append(meal_vector[idx])
            if len(meal_time) == 0:
                ug_plant = 0
            else:
                dg_plant = ((np.array(meal_amount) / weight) / 0.18)
                mt = np.array(meal_time)
                ug_plant = sum(
                    (dg_plant * ag * time_space * (idx - mt) * np.exp(-time_space * (idx - mt) / tmaxg)) / (tmaxg ** 2))
            ml_vec_plant = np.zeros(8)
            ml_vec_plant[0] = 1

            # excercise response 運動量の調整ができないため、ここの箇所は何もしない。今後修正するかもしれない
            m_e_pgu = 1
            m_e_piu = 1
            m_e_hgp = 1

            ap, bp, cp, dp = sh_glucoregulatory(
                xm_plant,
                fc01, vdg, k12, ag, tmaxg, egp0, tmaxi, ke, vdi, ka1, ka2, ka3, sf1, sf2, sf3,
                time_space, m_e_piu, m_e_pgu, m_e_hgp)
            xm_plant = (np.dot(ap, xm_plant).T + np.dot(bp, np.array(u_total)).T + dp.T + ml_vec_plant * (
                    ug_plant + ur_plant) * time_space).T  # y: mmol/kg
            y_plant = ((np.dot(cp, xm_plant) / vdg) * 18)[0]
            bg_output.append(float(y_plant))
            ins_input.append(float(u_total * weight * 60 / 1000))  # convert from mu/kg/min to u/hr

            if y_plant < (thr_resc) and cntr_resc == win_resc:

                if len(time_resc) == 0 or (len(time_resc) > 0 and idx - time_resc[-1] > 60 / time_space):
                    delay_resc = delay_rescue_val / time_space
                elif len(time_resc) > 0 and idx - time_resc[-1] <= win_resc:
                    delay_resc = 0 / time_space
                ins_adj_resc[int(idx + delay_resc): int(idx + delay_resc + timer_resc)] = iir_red_resc
                resc_trig_cntr = 1
                time_resc.append(idx + delay_resc)
                dr_plant = (carbs_resc / weight) / 0.18
                dr_mdl = (carbs_resc / weight) * 1000
                cntr_resc = 0

            if resc_trig_cntr == 1:
                ur_plant = (dr_plant * ag * time_space * (idx - np.array(time_resc) + 1) * np.exp(
                    -time_space * (idx - np.array(time_resc) + 1) / tmax_resc)) / (tmax_resc ** 2)
                ur_mdl = (dr_mdl * ag * time_space * (idx - np.array(time_resc) + 1) * np.exp(
                    -time_space * (idx - np.array(time_resc) + 1) / tmax_resc)) / (tmax_resc ** 2)
                ur_plant = np.where(ur_plant < 0, 0, ur_plant)
                ur_mdl = np.where(ur_mdl < 0, 0, ur_mdl)

            ur_plant = np.sum(ur_plant)
            ur_mdl = np.sum(ur_mdl)

            if idx > 0:
                if len(time_resc) > 0 and (idx - time_resc[-1]) >= win_resc - 1:
                    cntr_resc = win_resc

        return bg_output, ins_input

    def evaluate(self, simulation_days, weight, starting_glucose, meal_scenario, bg_true, time_space=5, _params=None):
        pred_time_space = 5
        bg_pred = self.predict(simulation_days, weight, starting_glucose, meal_scenario, _params)[0]
        label = []
        pred = []
        if time_space > pred_time_space:
            for j in range(len(bg_true)):
                label.append(bg_true[j])
                pred.append(bg_pred[int(j * pred_time_space / time_space)])
        else:
            for j in range(len(bg_pred)):
                label.append(bg_true[int(j * time_space / pred_time_space)])
                pred.append(bg_pred[j])
        mae = mean_absolute_error(label, pred)
        rmse = np.sqrt(mean_squared_error(label, pred))
        return mae, rmse

    def fit(
            self,
            simulation_days_list,
            weight_list,
            starting_glucose_list,
            meal_scenario_list,
            bg_true_list,
            time_space=5,
            method="GA",
            n=300):

        if method == "GA":
            from deap import base, tools, creator, algorithms

            def evaluate(individual):
                hidden_params = OrderedDict()
                for i, (k, v) in enumerate(self.param_set.items()):
                    hidden_params[k] = individual[i] * (v["max"] - v["min"]) + v["min"]
                mae_list = []
                try:
                    for i in range(len(meal_scenario_list)):
                        mae_list.append(
                            self.evaluate(
                                simulation_days_list[i],
                                weight_list[i],
                                starting_glucose_list[i],
                                meal_scenario_list[i],
                                bg_true_list[i],
                                time_space,
                                hidden_params)[0])
                except ValueError:
                    return 1.0e+300
                return np.mean(mae_list),

            def mutParams(individual, indpb):
                for i in range(len(individual)):
                    if random.random() < indpb:
                        individual[i] = random.random()
                return individual,

            def initPopulation(pcls, ind_init, n):
                init_list = []
                for i, (k, v) in enumerate(self.param_set.items()):
                    init_list.append((v["default"] - v["min"]) / (v["max"] - v["min"]))
                return pcls(ind_init(init_list) for x in range(n))

            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()
            toolbox.register("attribute", random.random)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, 24)
            # toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("population_guess", initPopulation, list, creator.Individual)
            toolbox.register("select", tools.selTournament, tournsize=5)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", mutParams, indpb=0.2)
            toolbox.register("evaluate", evaluate)

            #random.seed(64)

            NGEN = int(n / 150 + 1)
            POP = int(NGEN * 7.5)
            CXPB = 0.9
            MUTPB = 0.1

            pop = toolbox.population_guess(n=POP)
            for individual in pop:
                individual.fitness.values = toolbox.evaluate(individual)
            hof = tools.ParetoFront()

            print("GA: Generation={}, Population={}".format(NGEN, POP))
            algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, halloffame=hof, verbose=0)
            best_ind = tools.selBest(pop, 1)[0]
            best_value = best_ind.fitness.values

            best_params = OrderedDict()
            for i, (k, v) in enumerate(self.param_set.items()):
                best_params[k] = best_ind[i] * (v["max"] - v["min"]) + v["min"]
                self.params[k] = best_params[k]

            print("best params:", best_params)
            print("best value:", best_value)

        elif method == "BO":
            import optuna

            def objective(trial):
                hidden_params = OrderedDict()
                hidden_params["Fc01"] = trial.suggest_float("Fc01", self.param_set["Fc01"]["min"], self.param_set["Fc01"]["max"])
                hidden_params["VdG"] = trial.suggest_float("VdG", self.param_set["VdG"]["min"], self.param_set["VdG"]["max"])
                hidden_params["k12"] = trial.suggest_float("k12", self.param_set["k12"]["min"], self.param_set["k12"]["max"])
                hidden_params["Ag"] = trial.suggest_float("Ag", self.param_set["Ag"]["min"], self.param_set["Ag"]["max"])
                hidden_params["tmaxG"] = trial.suggest_float("tmaxG", self.param_set["tmaxG"]["min"], self.param_set["tmaxG"]["max"])
                hidden_params["EGP0"] = trial.suggest_float("EGP0", self.param_set["EGP0"]["min"], self.param_set["EGP0"]["max"])
                hidden_params["tmaxI"] = trial.suggest_float("tmaxI", self.param_set["tmaxI"]["min"], self.param_set["tmaxI"]["max"])
                hidden_params["Ke"] = trial.suggest_float("Ke", self.param_set["Ke"]["min"], self.param_set["Ke"]["max"])
                hidden_params["VdI"] = trial.suggest_float("VdI", self.param_set["VdI"]["min"], self.param_set["VdI"]["max"])
                hidden_params["ka1"] = trial.suggest_float("ka1", self.param_set["ka1"]["min"], self.param_set["ka1"]["max"])
                hidden_params["ka2"] = trial.suggest_float("ka2", self.param_set["ka2"]["min"], self.param_set["ka2"]["max"])
                hidden_params["ka3"] = trial.suggest_float("ka3", self.param_set["ka3"]["min"], self.param_set["ka3"]["max"])
                hidden_params["Sf1"] = trial.suggest_float("Sf1", self.param_set["Sf1"]["min"], self.param_set["Sf1"]["max"])
                hidden_params["Sf2"] = trial.suggest_float("Sf2", self.param_set["Sf2"]["min"], self.param_set["Sf2"]["max"])
                hidden_params["Sf3"] = trial.suggest_float("Sf3", self.param_set["Sf3"]["min"], self.param_set["Sf3"]["max"])
                hidden_params["TDIR_basal_rate"] = trial.suggest_float("TDIR_basal_rate", self.param_set["Fc01"]["min"], self.param_set["Fc01"]["max"])
                hidden_params["Ip"] = trial.suggest_float("Ip", self.param_set["Ip"]["min"], self.param_set["Ip"]["max"])
                hidden_params["tmax_resc"] = trial.suggest_float("tmax_resc", self.param_set["tmax_resc"]["min"], self.param_set["tmax_resc"]["max"])
                hidden_params["Thr_resc"] = trial.suggest_float("Thr_resc", self.param_set["Thr_resc"]["min"], self.param_set["Thr_resc"]["max"])
                hidden_params["Carbs_resc"] = trial.suggest_float("Carbs_resc", self.param_set["Carbs_resc"]["min"], self.param_set["Carbs_resc"]["max"])
                hidden_params["Win_resc"] = trial.suggest_float("Win_resc", self.param_set["Win_resc"]["min"], self.param_set["Win_resc"]["max"])
                hidden_params["IIR_red_resc"] = trial.suggest_float("IIR_red_resc", self.param_set["IIR_red_resc"]["min"], self.param_set["IIR_red_resc"]["max"])
                hidden_params["timer_resc"] = trial.suggest_float("timer_resc", self.param_set["timer_resc"]["min"], self.param_set["timer_resc"]["max"])
                hidden_params["delay_rescue_val"] = trial.suggest_float("delay_rescue_val", self.param_set["delay_rescue_val"]["min"], self.param_set["delay_rescue_val"]["max"])

                mae_list = []
                try:
                    for i in range(len(meal_scenario_list)):
                        mae_list.append(
                            self.evaluate(
                                simulation_days_list[i],
                                weight_list[i],
                                starting_glucose_list[i],
                                meal_scenario_list[i],
                                bg_true_list[i],
                                time_space,
                                hidden_params)[0])
                except ValueError:
                    return 1.0e+300
                return np.mean(mae_list)

            default_params = OrderedDict()
            for key, value in self.param_set.items():
                default_params[key] = value["default"]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                study = optuna.create_study(direction="minimize")
                study.enqueue_trial(dict(default_params))
                optuna.logging.disable_default_handler()
                study.optimize(objective, n_trials=n)

            self.params = OrderedDict(study.best_params)

            print("best params:", self.params)
            print("best value:", study.best_value)


def bg_plot(bg_output, ins_input):
    import matplotlib.pyplot as plt

    xaxis_time = np.arange(len(bg_output)) * 5 / 60 / 24

    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(xaxis_time, bg_output, "C0", label="Glucose [mg/dl]")
    ax2 = ax1.twinx()
    ax2.plot(xaxis_time, ins_input, "C1", label="Insulin [u/hr]")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left')

    ax1.set_xlabel("Day")
    ax1.set_ylabel("Glucose [mg/dl]")
    ax2.set_ylabel("Insulin [u/hr]")

    plt.show()


if __name__ == "__main__":
    model = SingleHormoneBloodGlucoseModel()

    import bgdata
    json_path = "../test/BG_dataset_20220204.json"
    dataset = bgdata.BGDataset.from_json(json_path)
    days_list = [10 for x in range(6)]
    weight_list = [53.2, 54.7, 70.5, 64.3, 68.0, 53.2]
    start_bg_list = []
    meal_list = []  # 分, 量
    bg_list = []  # 血糖値
    for paitient in dataset.paitients:
        bg_data = paitient.get_bg()
        ml_data = paitient.get_meal()
        bg_data = bgdata.compliment_bg(bg_data, 15, silent=True)
        bg_data, ml_data = bgdata.add_time_course(bg_data, ml_data)
        bg_data_2list, ml_data_2list = bgdata.split_days(bg_data, ml_data, 10)
        bg_data2 = bg_data_2list[0].loc[:, "blood_glucose_value"].to_numpy()
        ml_data2 = ml_data_2list[0].loc[:, ["time_course", "carbohydrate"]].to_numpy()
        starting_glucose = bg_data2[0]
        meal_scenario = ml_data2 * np.array([1440, 1])
        start_bg_list.append(starting_glucose)
        meal_list.append(meal_scenario)
        bg_list.append(bg_data2)
    # evaluate
    #for i in range(6):
    #    mae, rmse = model.evaluate(10, weight_list[i], start_bg_list[i], meal_list[i], bg_list[i], 15)
    # fit
    model.fit(days_list, weight_list, start_bg_list, meal_list, bg_list, 15, "GA", 1000)

    print(model.params)





