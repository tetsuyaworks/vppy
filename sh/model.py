from collections import OrderedDict
import numpy as np
from scipy.optimize import fsolve
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SingleHormoneBloodGlucoseModel:

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

    def predict(self, simulation_days, weight, starting_glucose, meal_scenario):

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
        for v in scenario:
            if v[0] > sim_time * time_space:
                continue
            meal_vector[round(v[0] / time_space)] = v[1]
        meal_time = []
        meal_amount = []

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
        glucose_set_point = 115
        tdir = calc_tdir(
            weight,
            fc01, vdg, k12, ag, tmaxg, egp0, tmaxi, ke, vdi, ka1, ka2, ka3, sf1, sf2, sf3,
            tdir_basal_rate, glucose_set_point
        )
        icr = (1700 / tdir / 3)
        ip = self.params["Ip"]
        bolus = ip * (meal_vector / icr) * 1000 / (weight * time_space)
        cgm_start = starting_glucose
        xm_plant = set_initial_conditions(glucose_set_point, fc01, vdg, k12, ag, tmaxg, egp0, tmaxi, ke, vdi, ka1, ka2,
                                          ka3,
                                          sf1, sf2, sf3)
        q1 = cgm_start / 18 * vdg
        xm_plant[0] = q1
        xm_plant = np.array([xm_plant]).T

        pgua_1_act = 0

        tmax_resc = self.params["tmax_resc"]
        thr_resc = self.params["Thr_resc"]
        carbs_resc = self.params["Carbs_resc"]
        win_resc = self.params["Win_resc"] / time_space
        iir_red_resc = self.params["IIR_red_resc"]
        timer_resc = self.params["timer_resc"] / time_space
        delay_rescue_val = self.params["delay_rescue_val"]
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

    def evaluate(self, simulation_days, weight, starting_glucose, meal_scenario, bg_true, time_space=5):
        pred_time_space = 5
        bg_pred = self.predict(simulation_days, weight, starting_glucose, meal_scenario)[0]
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

    def fit(self, simulation_days, weight, starting_glucose, meal_scenario, bg_true):
        pass  # パラメータ最適化を遺伝的アルゴリズムか確率的勾配降下法のどちらかで実装する予定


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

    scenario = [
        # Time of meal event (min)
        [40, 250, 605, 1515, 1985, 2020, 2165, 2830, 3225, 3515, 4355, 4570, 4925],
        # gramAmount carbs in meal event (gram)
        [35, 79, 117, 40, 15, 100, 30, 100, 100, 100, 35, 79, 117]]
    scenario = np.array(scenario).T

    #bg_output, ins_input = model.predict(1, 45.0, 160, scenario)
    #bg_plot(bg_output, ins_input)

    import random
    bg_true = [random.randint(30, 400) for x in range(96)]

    #m, r = model.evaluate(1, 45.0, 160, scenario, bg_true, 15)
    #print(m, r)



