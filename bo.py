import os
import sys # 用于退出程序
import time
import itertools
import openpyxl # 导入 openpyxl 库，用于处理 Excel 文件
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # 导入 MinMaxScaler 进行数据归一化
from scipy.spatial import distance # 导入 distance 模块计算距离
from scipy.stats import norm, qmc # 导入 norm 用于计算正态分布相关函数, qmc 用于 Sobol 序列
import GPy # 导入 GPy 库，用于高斯过程建模
import cma # 导入 CMA-ES 算法库

# --- 默认参数设置 ---
# 这些参数部分可以通过用户输入进行覆盖
DEFAULT_OUTPUT_LABEL = "Score"      # BO_Exp_Table.xlsx 中的输出标签, 要优化的输出变量名
DEFAULT_XI_INITIAL = 0.1           # 探索-利用权衡参数 (EI采集函数)，初始值。调整为较小值，如果EI过大。
DEFAULT_XI_DECAY_FACTOR = 0.95      # XI衰减因子，每次迭代后 xi *= xi_decay_factor
DEFAULT_MIN_XI = 0.01              # XI的最小值
DEFAULT_GP_OPTIMIZER_RESTARTS = 10  # GP模型超参数优化的重启次数
DEFAULT_CORE_OPT_MAX_ITERS = 1000   # 高斯过程模型内部优化的最大迭代次数
DEFAULT_MIN_DIST_TO_EXISTING = 0.01 # (归一化空间) 排除与所有已知"原始"实验点过近的候选点，欧氏距离阈值
DEFAULT_EI_CONVERGENCE_THRESHOLD = 1e-7 # 当最大EI低于此阈值时，可能提前终止当前优化周期的建议 (调低阈值)
DEFAULT_PREDICTED_VALUE_STAGNATION_THRESHOLD = 1e-4 # 预测最优值停滞阈值
DEFAULT_STAGNATION_PATIENCE = 3     # 预测最优值连续停滞N次后，可能提前终止
DEFAULT_WHITE_NOISE_VARIANCE = 1e-5 # 为GP模型添加的白噪声方差
MAX_RANDOM_FALLBACK_ATTEMPTS = 100 # CMA-ES卡住时，随机采样回退的最大尝试次数

# --- 函数定义 ---

def get_user_input_for_sobol():
    """获取用户输入的用于生成Sobol初始实验设计的信息"""
    print("\n--- [初始实验设计 Sobol 序列生成] ---")
    while True:
        try:
            n_factors = int(input("请输入因子(变量)的数量 (例如: 3): "))
            if n_factors > 0:
                break
            else:
                print("[输入错误] 因子数量必须是正整数。")
        except ValueError:
            print("[输入错误] 请输入一个有效的整数。")

    factor_names = []
    factor_bounds = []
    print("接下来，请输入每个因子的名称及其取值范围 (最小值和最大值)。")
    for i in range(n_factors):
        while True:
            name = input(f"  请输入第 {i+1} 个因子的名称 (例如: Temperature): ").strip()
            if name:
                factor_names.append(name)
                break
            else:
                print("[输入错误] 因子名称不能为空。")
        while True:
            try:
                min_val = float(input(f"  请输入因子 '{name}' 的最小值 (例如: 50): "))
                max_val = float(input(f"  请输入因子 '{name}' 的最大值 (例如: 100): "))
                if max_val > min_val:
                    factor_bounds.append((min_val, max_val))
                    break
                else:
                    print("[输入错误] 最大值必须大于最小值。")
            except ValueError:
                print("[输入错误] 请输入有效的数值。")

    while True:
        try:
            num_sobol_points = int(input(f"请输入希望生成的 Sobol 初始实验点数量 (建议至少 {n_factors*5}): "))
            if num_sobol_points > 0:
                break
            else:
                print("[输入错误] 实验点数量必须是正整数。")
        except ValueError:
            print("[输入错误] 请输入一个有效的整数。")
    return factor_names, factor_bounds, num_sobol_points

def generate_sobol_doe(factor_names, factor_bounds, num_points, output_col_name, filename="BO_Exp_Table.xlsx"):
    """使用Sobol序列生成初始实验设计并保存到Excel文件"""
    n_dims = len(factor_names)
    sampler = qmc.Sobol(d=n_dims, scramble=True)
    sobol_points_normalized = sampler.random(n=num_points) # 生成 [0,1] 范围内的点

    # 将归一化的点转换到实际尺度
    sobol_points_scaled = np.zeros_like(sobol_points_normalized)
    min_vals = np.array([b[0] for b in factor_bounds])
    max_vals = np.array([b[1] for b in factor_bounds])
    for i in range(n_dims):
        sobol_points_scaled[:, i] = qmc.scale(sobol_points_normalized[:, i].reshape(-1, 1), min_vals[i], max_vals[i]).flatten()

    # MODIFIED: 将生成的因子值四舍五入到两位小数
    sobol_points_scaled = np.round(sobol_points_scaled, 2)

    # 创建DataFrame
    df_data = {name: sobol_points_scaled[:, i] for i, name in enumerate(factor_names)}
    df = pd.DataFrame(df_data)
    df[output_col_name] = "PENDING_USER_INPUT"

    # 创建包含MIN/MAX的完整DataFrame
    header_df = pd.DataFrame(columns=factor_names + [output_col_name])
    min_row = {name: factor_bounds[i][0] for i, name in enumerate(factor_names)}
    min_row[output_col_name] = np.nan # 或者其他标记
    max_row = {name: factor_bounds[i][1] for i, name in enumerate(factor_names)}
    max_row[output_col_name] = np.nan

    # 使用 concat 来添加 MIN/MAX 行，并确保索引正确
    final_df = pd.concat([
        pd.DataFrame(min_row, index=["MIN"], columns=header_df.columns),
        pd.DataFrame(max_row, index=["MAX"], columns=header_df.columns),
        df
    ], axis=0)
    final_df.index.name = "Experiment_ID"


    try:
        final_df.to_excel(filename)
        print(f"\n[成功] Sobol 初始实验设计已生成并保存到 '{filename}'。")
        print(f"       因子值已四舍五入到两位小数。") # 新增提示
        print(f"       请打开文件，完成 '{output_col_name}' 列的实验结果填写。")
        print("       完成后，请重新运行此脚本以开始贝叶斯优化。")
    except Exception as e:
        print(f"[错误] 保存文件 '{filename}' 失败: {e}")
    sys.exit() # 生成初始设计后退出

def expected_improvement(points_norm, gp_model, y_max_observed_norm, xi, normalizer_y_std_original):
    """
    计算给定点的预期提升(EI)
    points_norm: 归一化后的输入点
    gp_model: 训练好的GPy高斯过程模型
    y_max_observed_norm: 当前观测到的Y的最大值 (在归一化空间)
    xi: 探索-利用平衡参数
    normalizer_y_std_original: Y值在原始尺度上的标准差 (用于可选的EI反归一化，当前未使用)
    """
    if points_norm.ndim == 1:
        points_norm = points_norm.reshape(1, -1)

    pred_mean_norm, pred_var_norm = gp_model.predict_noiseless(points_norm) 
    
    pred_std_norm = np.sqrt(np.maximum(pred_var_norm, 1e-18)) # 确保方差非负且标准差不会太小

    imp = pred_mean_norm - y_max_observed_norm - xi 
    Z = np.zeros_like(imp)
    valid_std_mask = pred_std_norm > 1e-9 # 定义一个阈值来判断标准差是否有效
    if np.any(valid_std_mask):
        Z[valid_std_mask] = imp[valid_std_mask] / pred_std_norm[valid_std_mask]
    
    ei_norm = imp * norm.cdf(Z) + pred_std_norm * norm.pdf(Z)
    # 当标准差极小时，EI主要由提升量决定（如果提升为正）
    ei_norm[~valid_std_mask] = np.maximum(0, imp[~valid_std_mask]) 
    
    return ei_norm.flatten()


def optimize_acquisition_cmaes(gp_model, y_max_observed_norm, xi, bounds_norm, n_dim, normalizer_y_std_original, current_all_known_points_norm, initial_reject_rad_norm_for_suggestions):
    """使用CMA-ES优化采集函数 (目标是最小化 -EI)"""
    
    # 引入一个动态的局部拒绝半径，用于此函数调用内的尝试
    local_reject_rad = initial_reject_rad_norm_for_suggestions
    reject_rad_decay_factor = 0.9 # 局部拒绝半径的衰减因子
    max_local_reject_attempts = 5 # 尝试放松局部拒绝半径的次数

    def objective_function(x_norm, current_local_reject_rad):
        x_norm_clipped = np.clip(x_norm, 0, 1)
        
        # 如果当前已知点存在，则对过于接近已知点的点施加惩罚以引导 CMA-ES
        if current_all_known_points_norm.size > 0:
            # 如果 x_norm_clipped 是 1D 数组，则将其重塑为 2D 以便 cdist 使用
            if x_norm_clipped.ndim == 1:
                x_norm_clipped_reshaped = x_norm_clipped.reshape(1, -1)
            else:
                x_norm_clipped_reshaped = x_norm_clipped

            dist_to_known = distance.cdist(x_norm_clipped_reshaped, current_all_known_points_norm)
            if dist_to_known.min() < current_local_reject_rad: # 使用动态的 local_reject_rad
                return np.inf # 施加重罚
        
        ei = expected_improvement(np.array(x_norm_clipped), gp_model, y_max_observed_norm, xi, normalizer_y_std_original)
        if ei.size == 0 or np.isnan(ei[0]) or np.isinf(ei[0]): # 增加对无效EI值的检查
            return np.inf # 如果EI无效，返回一个很大的值，CMA-ES会避免这个区域
        return -ei[0] 

    x0 = np.random.rand(n_dim) 
    sigma0 = 0.3 
    options = {'bounds': [0, 1], 'verb_disp': 0, 'verbose': -9, 'maxiter': 100 * n_dim, 'tolfun': 1e-7} 

    for attempt_idx in range(max_local_reject_attempts):
        try:
            # 将当前的 local_reject_rad 传递给目标函数
            x_best_norm, es_instance = cma.fmin2(lambda x: objective_function(x, local_reject_rad), x0, sigma0, options=options)
            neg_ei_at_best = es_instance.result.fbest 
            
            # 检查 neg_ei_at_best 是否有效
            if np.isnan(neg_ei_at_best) or np.isinf(neg_ei_at_best):
                raise ValueError("CMA-ES返回了无效的EI值")

            x_best_norm = np.clip(x_best_norm, 0, 1)
            
            if current_all_known_points_norm.size > 0:
                dist_to_known_final = distance.cdist(x_best_norm.reshape(1,-1), current_all_known_points_norm).min()
                if dist_to_known_final < initial_reject_rad_norm_for_suggestions: # Check against the original stricter threshold
                    print(f"    CMA-ES找到点 (距离 {dist_to_known_final:.4f}) 仍低于初始阈值 ({initial_reject_rad_norm_for_suggestions:.4f})。尝试放松距离限制。")
                    local_reject_rad *= reject_rad_decay_factor
                    x0 = np.random.rand(n_dim) # 为下一次尝试重置 x0
                    continue # 使用放松的 local_reject_rad 再次尝试
            
            return x_best_norm, -neg_ei_at_best 
        except Exception as e:
            print(f"[警告] CMA-ES 优化采集函数时出错 (尝试 {attempt_idx+1}/{max_local_reject_attempts}): {e}")
            print(f"       当前局部拒绝半径: {local_reject_rad:.4f}。尝试放松距离限制。")
            local_reject_rad *= reject_rad_decay_factor
            x0 = np.random.rand(n_dim) # 为下一次尝试重置 x0
            continue # 使用放松的 local_reject_rad 再次尝试

    print("[警告] CMA-ES 在多次尝试放松距离限制后仍未能找到合适的点。将尝试使用随机点作为候补。")
    # 如果 CMA-ES 在多次尝试后失败，则回退到随机点
    for _ in range(MAX_RANDOM_FALLBACK_ATTEMPTS):
        fallback_x = np.random.rand(n_dim)
        if current_all_known_points_norm.size > 0:
            dist_to_known_rand = distance.cdist(fallback_x.reshape(1,-1), current_all_known_points_norm)
            if dist_to_known_rand.min() < local_reject_rad: # 回退时使用最终放松的 local_reject_rad
                continue
        fallback_ei_array = expected_improvement(fallback_x, gp_model, y_max_observed_norm, xi, normalizer_y_std_original)
        if fallback_ei_array.size > 0 and not (np.isnan(fallback_ei_array[0]) or np.isinf(fallback_ei_array[0])):
            return fallback_x, fallback_ei_array[0]
    print("[警告] 随机点回退也未能找到有效的EI值。")
    return np.random.rand(n_dim), -np.inf # 返回一个随机点和极小的EI


# --- 主程序 ---
if __name__ == "__main__":
    print("--- 贝叶斯优化脚本 ---")

    while True:
        choice = input("您是否希望生成一个新的初始实验设计 (使用Sobol序列) 并保存到 'BO_Exp_Table.xlsx'? (y/n): ").strip().lower()
        if choice in ['y', 'n']:
            break
        print("[输入错误] 请输入 'y' 或 'n'.")

    if choice == 'y':
        factor_info_names, factor_info_bounds, num_sobol = get_user_input_for_sobol()
        generate_sobol_doe(factor_info_names, factor_info_bounds, num_sobol, DEFAULT_OUTPUT_LABEL, "BO_Exp_Table.xlsx")

    print("\n--- [贝叶斯优化流程开始] ---")
    print("确保 'BO_Exp_Table.xlsx' 文件存在，并且包含 MIN/MAX 定义行以及至少一些初始实验数据和结果。")
    print("此脚本的建议点适用于并行实验。") 

    output_label = DEFAULT_OUTPUT_LABEL 

    while True:
        try:
            user_input = input(f"请输入每个优化周期建议的实验数量 (正整数, 例如: 5): ")
            number_of_experiments_per_cycle = int(user_input)
            if number_of_experiments_per_cycle > 0:
                print(f"[参数设置] 每个优化周期建议的实验数量将设置为: {number_of_experiments_per_cycle}")
                break
            else:
                print("[输入错误] 实验数量必须是正整数，请重新输入。")
        except ValueError:
            print("[输入错误] 请输入一个有效的整数。")

    while True:
        try:
            user_input_daf = input(f"请输入建议点间最小距离的疏密调整因子 (建议 0.2-0.6, 较小值点更密集, 例如: 0.35): ")
            distance_adjustment_factor = float(user_input_daf)
            if 0.1 <= distance_adjustment_factor <= 1.0: 
                 print(f"[参数设置] 建议点间最小距离的疏密调整因子将设置为: {distance_adjustment_factor}")
                 break
            else:
                print("[输入错误] 调整因子建议在0.1到1.0之间。")
        except ValueError:
            print("[输入错误] 请输入一个有效的数值。")

    xi_initial = DEFAULT_XI_INITIAL
    xi_decay_factor = DEFAULT_XI_DECAY_FACTOR
    min_xi = DEFAULT_MIN_XI
    gp_optimizer_restarts = DEFAULT_GP_OPTIMIZER_RESTARTS 
    core_opt_max_iters = DEFAULT_CORE_OPT_MAX_ITERS
    min_dist_to_existing_norm = DEFAULT_MIN_DIST_TO_EXISTING 
    ei_convergence_threshold = DEFAULT_EI_CONVERGENCE_THRESHOLD
    predicted_value_stagnation_threshold = DEFAULT_PREDICTED_VALUE_STAGNATION_THRESHOLD
    stagnation_patience = DEFAULT_STAGNATION_PATIENCE
    white_noise_variance = DEFAULT_WHITE_NOISE_VARIANCE

    if os.path.exists("./GP_BO_Results"):
        num = 1
        while True:
            backup_name = f"./GP_BO_Results_prev{num}"
            if not os.path.exists(backup_name):
                try:
                    os.rename("./GP_BO_Results", backup_name)
                    print(f"[目录管理] 已将现有 'GP_BO_Results' 目录备份为 '{backup_name}'")
                except Exception as e:
                    print(f"[警告] 备份目录失败: {e}。结果仍将保存在 GP_BO_Results 中。")
                break
            num += 1
    os.makedirs("./GP_BO_Results", exist_ok=True) 

    try:
        exp_table_full = pd.read_excel("./BO_Exp_Table.xlsx", index_col=0)
        exp_table_full.columns = [c.strip() for c in exp_table_full.columns] 
        if output_label not in exp_table_full.columns:
            print(f"[错误] 输出标签 '{output_label}' 在 'BO_Exp_Table.xlsx' 文件中未找到。请检查列名。")
            print(f"       可用列名: {exp_table_full.columns.tolist()}")
            sys.exit()
    except FileNotFoundError:
        print("[错误] 未找到 'BO_Exp_Table.xlsx' 文件。请确保文件存在，或先运行初始实验设计生成步骤。")
        sys.exit()
    except Exception as e:
        print(f"[错误] 读取 'BO_Exp_Table.xlsx' 文件失败: {e}")
        sys.exit()

    x_data_column = [c for c in exp_table_full.columns if c != output_label]
    if not x_data_column:
        print("[错误] 未能从 'BO_Exp_Table.xlsx' 中识别任何输入因子列。")
        sys.exit()
    print(f"[数据读取] 因子: {x_data_column}")
    n_dimensions = len(x_data_column)

    try:
        min_li = [exp_table_full.loc["MIN", c] for c in x_data_column]
        max_li = [exp_table_full.loc["MAX", c] for c in x_data_column]
    except KeyError:
        print("[错误] 'BO_Exp_Table.xlsx' 文件中必须包含 'MIN' 和 'MAX' 行来定义各因子的范围。")
        sys.exit()

    min_max_li_for_scaler = np.array([min_li, max_li], dtype=float)
    mmscaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    mmscaler.fit(min_max_li_for_scaler)

    exp_table_initial_with_pending = exp_table_full.drop(["MIN", "MAX"], errors='ignore').copy() 
    if output_label in exp_table_initial_with_pending.columns:
        exp_table_initial_with_pending[output_label] = exp_table_initial_with_pending[output_label].astype(object)
    exp_table_initial_with_pending[output_label] = pd.to_numeric(exp_table_initial_with_pending[output_label], errors='coerce')
    
    initial_completed_exp = exp_table_initial_with_pending.dropna(subset=[output_label]).copy()
    
    if len(initial_completed_exp) < (n_dimensions + 1): 
        print(f"[警告] 已完成的初始实验数据量 ({len(initial_completed_exp)} 点) 过少 (少于 维度+1 = {n_dimensions+1} 点)。")
        print("       强烈建议在 'BO_Exp_Table.xlsx' 中提供更多带有实际结果的初始实验点以建立可靠的初始模型。")
        print("       如果继续，模型可能非常不可靠。")
        if input("是否仍要继续? (y/n): ").lower() != 'y':
            sys.exit()
    
    if number_of_experiments_per_cycle > 1 and n_dimensions > 0:
        characteristic_spacing_factor = (1.0 / number_of_experiments_per_cycle)**(1.0 / n_dimensions)
        reject_rad_norm_for_suggestions = distance_adjustment_factor * characteristic_spacing_factor
    elif number_of_experiments_per_cycle == 1 and n_dimensions > 0:
        characteristic_spacing_factor = (1.0 / 2.0)**(1.0 / n_dimensions)
        reject_rad_norm_for_suggestions = distance_adjustment_factor * characteristic_spacing_factor
    elif n_dimensions == 0 : 
         reject_rad_norm_for_suggestions = 0.01 
    else: 
        reject_rad_norm_for_suggestions = distance_adjustment_factor * 0.1 

    reject_rad_norm_for_suggestions = np.clip(reject_rad_norm_for_suggestions, 0.01, 0.4)
    
    print(f"[参数] 本轮建议点间最小目标距离 (reject_rad_norm_for_suggestions, 归一化空间): {reject_rad_norm_for_suggestions:.4f}")
    print(f"[参数] 与历史数据点最小距离 (min_dist_to_existing_norm, 归一化空间): {min_dist_to_existing_norm:.4f}")


    print("\n--- [准备基础高斯过程模型 (基于初始真实数据)] ---")
    if len(initial_completed_exp) == 0:
        print("[错误] 没有已完成的初始实验数据用于训练基础GP模型。请检查 'BO_Exp_Table.xlsx'。")
        sys.exit()
        
    x_train_original_init = initial_completed_exp.loc[:, x_data_column].values
    y_train_original_init = initial_completed_exp.loc[:, [output_label]].values 
    x_train_norm_init = mmscaler.transform(x_train_original_init)

    kernel_init = GPy.kern.Matern52(input_dim=n_dimensions, ARD=True) + GPy.kern.White(input_dim=n_dimensions, variance=white_noise_variance)
    kernel_init.white.variance.constrain_bounded(1e-7, 1e-3) 
    
    base_gp_model = GPy.models.GPRegression(X=x_train_norm_init, Y=y_train_original_init, kernel=kernel_init, normalizer=True)
    
    optimization_successful = False
    try:
        print(f"正在优化基础GP模型超参数 (使用 optimize_restarts, 重启次数: {gp_optimizer_restarts})...")
        base_gp_model.optimize_restarts(num_restarts=gp_optimizer_restarts, 
                                        optimizer='lbfgsb', 
                                        max_iters=core_opt_max_iters, 
                                        verbose=False, 
                                        messages=False) 
        print("基础GP模型优化完成。")
        optimization_successful = True
    except Exception as e:
        print(f"[警告] 基础GP模型优化失败: {e}。将使用当前模型参数。")

    print("\n--- [基础GP模型参数详细信息] ---")
    try:
        if hasattr(base_gp_model, 'kern') and hasattr(base_gp_model.kern, 'Mat52'):
             if hasattr(base_gp_model.kern.Mat52, 'variance'):
                 print(f"参数: kern.Mat52.variance, 值: {base_gp_model.kern.Mat52.variance.values.tolist()}")
             if hasattr(base_gp_model.kern.Mat52, 'lengthscale'):
                 print(f"参数: kern.Mat52.lengthscale, 值: {base_gp_model.kern.Mat52.lengthscale.values.tolist()}")
        if hasattr(base_gp_model, 'kern') and hasattr(base_gp_model.kern, 'white'):
            if hasattr(base_gp_model.kern.white, 'variance'): 
                print(f"参数: kern.white.variance, 值: {base_gp_model.kern.white.variance.values.tolist()}")
        if hasattr(base_gp_model, 'Gaussian_noise'): 
            if hasattr(base_gp_model.Gaussian_noise, 'variance'):
                print(f"参数: Gaussian_noise.variance, 值: {base_gp_model.Gaussian_noise.variance.values.tolist()}")
        
        print(f"完整参数数组 (model.param_array): {base_gp_model.param_array.tolist()}")
        print(f"模型概览 (print(model)):")
        print(base_gp_model)

    except Exception as e_print:
        print(f"打印参数时出错: {e_print}")
    print("--- [模型参数结束] ---\n")
    
    # MODIFIED: More stable extraction of Y normalization parameters (fixed in previous turn)
    stable_y_mean_for_unnorm = 0.0
    stable_y_std_for_unnorm = 1.0
    params_found_method = "默认值(std=1, mean=0) (未找到GP模型归一化器或训练数据)"

    if base_gp_model.normalizer is not None:
        try:
            accessor_mean = base_gp_model.normalizer.mean
            accessor_std = base_gp_model.normalizer.std

            val_mean = accessor_mean() if callable(accessor_mean) else accessor_mean
            val_std = accessor_std() if callable(accessor_std) else accessor_std
            
            if isinstance(val_mean, np.ndarray):
                stable_y_mean_for_unnorm = float(val_mean.item()) if val_mean.size == 1 else float(np.mean(val_mean))
            elif isinstance(val_mean, (int, float, np.number)):
                stable_y_mean_for_unnorm = float(val_mean)
            else:
                raise TypeError(f"无法将 normalizer 的均值转换为浮点数，类型为: {type(val_mean)}")

            if isinstance(val_std, np.ndarray):
                stable_y_std_for_unnorm = float(val_std.item()) if val_std.size == 1 else float(np.mean(val_std))
            elif isinstance(val_std, (int, float, np.number)):
                stable_y_std_for_unnorm = float(val_std)
            else:
                raise TypeError(f"无法将 normalizer 的标准差转换为浮点数，类型为: {type(val_std)}")

            if stable_y_std_for_unnorm < 1e-9: 
                print(f"[信息] 从GP模型 normalizer 获取的Y标准差 ({stable_y_std_for_unnorm:.2e}) 过小。反标准化std将设为1.0。")
                stable_y_std_for_unnorm = 1.0
            params_found_method = "GP_model.normalizer.mean/std"
        except Exception as e_norm:
            print(f"[警告] 从GP模型 normalizer 提取归一化参数失败: {e_norm}。将尝试使用原始数据统计。")
            if y_train_original_init.size > 0:
                stable_y_mean_for_unnorm = np.mean(y_train_original_init)
                stable_y_std_for_unnorm = np.std(y_train_original_init)
                if stable_y_std_for_unnorm < 1e-9:
                    print(f"[信息] 初始训练数据 y_train_original_init 的标准差 ({stable_y_std_for_unnorm:.2e}) 过小。反标准化std将设为1.0。")
                    stable_y_std_for_unnorm = 1.0
                params_found_method = "y_train_original_init的统计值 (GP normalizer提取失败后回退)"
    elif y_train_original_init.size > 0: 
        stable_y_mean_for_unnorm = np.mean(y_train_original_init)
        stable_y_std_for_unnorm = np.std(y_train_original_init)
        if stable_y_std_for_unnorm < 1e-9:
            print(f"[信息] 初始训练数据 y_train_original_init 的标准差 ({stable_y_std_for_unnorm:.2e}) 过小。反标准化std将设为1.0。")
            stable_y_std_for_unnorm = 1.0
        params_found_method = "y_train_original_init的统计值 (无GP normalizer)"
    
    print(f"[信息] 用于反标准化的稳定参数 (来源: {params_found_method}): mean={stable_y_mean_for_unnorm:.4f}, std={stable_y_std_for_unnorm:.4f}")

    print("\n--- [调试信息：基础GP模型标准化器状态] ---")
    if base_gp_model.normalizer is not None:
        print(f"基础模型 Y_normalizer 对象: {base_gp_model.normalizer}")
        # Try to print the actual mean/std values from the normalizer if they are attributes
        _norm_mean_val = base_gp_model.normalizer.mean() if callable(base_gp_model.normalizer.mean) else base_gp_model.normalizer.mean
        _norm_std_val = base_gp_model.normalizer.std() if callable(base_gp_model.normalizer.std) else base_gp_model.normalizer.std
        print(f"基础模型 normalizer.mean (获取值): {_norm_mean_val}")
        print(f"基础模型 normalizer.std (获取值): {_norm_std_val}")

    else:
        print("基础模型 normalizer 为 None。")
    
    if hasattr(base_gp_model, 'Y_normalized') and base_gp_model.Y_normalized is not None:
        print(f"基础模型 Y_normalized (前5个): {base_gp_model.Y_normalized[:5].flatten().tolist()}")
        y_max_observed_norm_init = np.max(base_gp_model.Y_normalized)
    else:
        print("基础模型没有 Y_normalized 属性或其为 None。将基于原始Y和提取的参数计算标准化的最大值。")
        if y_train_original_init.size > 0 and stable_y_std_for_unnorm > 1e-9 : 
            y_max_observed_norm_init = np.max((y_train_original_init - stable_y_mean_for_unnorm) / stable_y_std_for_unnorm)
        elif y_train_original_init.size > 0 : 
             y_max_observed_norm_init = np.max(y_train_original_init - stable_y_mean_for_unnorm) 
        else: 
            y_max_observed_norm_init = 0.0
    print(f"基础模型 y_max_observed_norm_init (用于EI计算): {y_max_observed_norm_init:.4f}")
    print("--- [调试信息结束] ---\n")

    start_time = time.time()
    current_xi = xi_initial
    
    exp_table_for_bo_loop = exp_table_initial_with_pending.copy() 
    if output_label in exp_table_for_bo_loop.columns:
        exp_table_for_bo_loop[output_label] = exp_table_for_bo_loop[output_label].astype(object)
    else: 
        exp_table_for_bo_loop[output_label] = pd.Series(dtype=object, index=exp_table_for_bo_loop.index) # 确保索引对齐

    suggested_points_in_this_bo_run_norm = [] 

    for i_cycle in range(1, number_of_experiments_per_cycle + 1):
        print(f"\n--- [贝叶斯优化建议 #{i_cycle}/{number_of_experiments_per_cycle}] XI: {current_xi:.4f} ---")
        print(f"正在使用CMA-ES优化采集函数 (基于基础GP模型)...")
        
        cma_attempts = 0
        max_cma_attempts = 10 
        next_point_norm = None
        max_ei_value = -np.inf
        xi_for_cma_attempt = current_xi

        while cma_attempts < max_cma_attempts:
            cma_attempts += 1
            print(f"  CMA-ES 尝试 #{cma_attempts} (使用 XI: {xi_for_cma_attempt:.4f})...")
            
            all_known_points_norm = list(x_train_norm_init) + suggested_points_in_this_bo_run_norm
            all_known_points_norm_np = np.array(all_known_points_norm) if all_known_points_norm else np.empty((0, n_dimensions))

            temp_next_point_norm, temp_max_ei_value = optimize_acquisition_cmaes(
                base_gp_model, y_max_observed_norm_init, xi_for_cma_attempt, 
                bounds_norm=[(0,1)] * n_dimensions, 
                n_dim=n_dimensions,
                normalizer_y_std_original = stable_y_std_for_unnorm,
                current_all_known_points_norm = all_known_points_norm_np, 
                initial_reject_rad_norm_for_suggestions = reject_rad_norm_for_suggestions
            )
            
            if temp_next_point_norm is None: # CMA-ES or fallback might return None if truly stuck
                print(f"    CMA-ES/Fallback未能提供有效的建议点 (尝试 {cma_attempts})。")
                if cma_attempts < max_cma_attempts:
                    print("    增加XI并尝试重新优化...")
                    xi_for_cma_attempt = min(xi_for_cma_attempt * 1.5, 0.5)
                    continue
                else:
                    print(f"    CMA-ES 在 {max_cma_attempts} 次尝试后仍未提供有效点。")
                    break # Exit CMA attempts loop

            min_dist_found = np.inf # Default if no known points to compare against
            if all_known_points_norm_np.size > 0:
                dist_to_known = distance.cdist(temp_next_point_norm.reshape(1,-1), all_known_points_norm_np)
                min_dist_found = dist_to_known.min()

            if min_dist_found < reject_rad_norm_for_suggestions and i_cycle > 1 : # Check distance, more critical for points after the first in a batch
                print(f"    CMA-ES建议点与已知点/本轮已建议点过近 (距离 {min_dist_found:.4f} < 目标阈值 {reject_rad_norm_for_suggestions:.4f})。")
                if cma_attempts < max_cma_attempts:
                    print("    增加XI并尝试重新优化...")
                    xi_for_cma_attempt = min(xi_for_cma_attempt * 1.5, 0.5) 
                    temp_next_point_norm = None # Discard this point and retry
                    continue 
                else: 
                    print(f"    CMA-ES 在 {max_cma_attempts} 次尝试后仍建议过近的点。尝试随机回退...")
                    random_fallback_success = False
                    for _ in range(MAX_RANDOM_FALLBACK_ATTEMPTS):
                        rand_point_norm = np.random.rand(n_dimensions)
                        dist_to_known_rand_check = np.inf
                        if all_known_points_norm_np.size > 0:
                           dist_to_known_rand_check = distance.cdist(rand_point_norm.reshape(1,-1), all_known_points_norm_np).min()
                        
                        if dist_to_known_rand_check >= reject_rad_norm_for_suggestions:
                            temp_next_point_norm = rand_point_norm
                            temp_max_ei_value_array = expected_improvement(temp_next_point_norm, base_gp_model, y_max_observed_norm_init, current_xi, stable_y_std_for_unnorm)
                            temp_max_ei_value = temp_max_ei_value_array[0] if temp_max_ei_value_array.size > 0 else -np.inf
                            print(f"    随机回退成功，选择点: {np.round(temp_next_point_norm,4)}, EI: {temp_max_ei_value:.6g}")
                            random_fallback_success = True
                            break
                    if not random_fallback_success:
                        print(f"    随机回退在 {MAX_RANDOM_FALLBACK_ATTEMPTS} 次尝试后也失败。标记本轮建议失败。")
                        temp_next_point_norm = None 
                    # Fall through to assign temp_next_point_norm to next_point_norm
            
            next_point_norm = temp_next_point_norm # Assign the found point (or None if all attempts failed)
            # If next_point_norm is not None, temp_max_ei_value should correspond to it.
            # If next_point_norm became None due to fallback failure, max_ei_value remains from previous valid attempt or -inf.
            # Ensure max_ei_value is updated only if temp_next_point_norm is valid
            if next_point_norm is not None:
                 max_ei_value = temp_max_ei_value # temp_max_ei_value should be from the successful CMA-ES or random fallback
            else:
                 max_ei_value = -np.inf # Explicitly set to -inf if no point was found

            break # Exit CMA-ES attempts loop
        
        if next_point_norm is None: 
            print(f"[警告] 未能找到合适的建议点。跳过本轮建议。")
            continue 

        print(f"最终选择点 (归一化): {np.round(next_point_norm,4)}, 最大 EI: {max_ei_value:.6g}")
        suggested_points_in_this_bo_run_norm.append(next_point_norm) 

        if max_ei_value < ei_convergence_threshold and not np.isinf(max_ei_value) and i_cycle > 1 : 
            print(f"[终止条件满足] 最大预期提升 ({max_ei_value:.2e}) 低于阈值 ({ei_convergence_threshold:.1e})。")
            print("       后续建议可能无显著改善。提前结束当前优化周期。")
            break
        
        next_point_original_scale = mmscaler.inverse_transform(next_point_norm.reshape(1, -1)).flatten()
        new_experiment_data = {}
        for idx, col_name in enumerate(x_data_column):
            # 将建议的因子值四舍五入到两位小数
            new_experiment_data[col_name] = round(next_point_original_scale[idx], 2)

        pred_mean_norm_at_next, _ = base_gp_model.predict_noiseless(next_point_norm.reshape(1,-1))
        pred_mean_original_at_next = pred_mean_norm_at_next[0,0] * stable_y_std_for_unnorm + stable_y_mean_for_unnorm
        
        new_experiment_data[output_label] = round(pred_mean_original_at_next, 2)

        new_index = f"BO_Cycle{i_cycle}_Sugg" 
        temp_idx_counter = 1
        final_new_index = new_index
        while final_new_index in exp_table_for_bo_loop.index:
            final_new_index = f"{new_index}_{temp_idx_counter}"
            temp_idx_counter += 1
        
        new_row_df = pd.DataFrame(new_experiment_data, index=[final_new_index])
        exp_table_for_bo_loop = pd.concat([exp_table_for_bo_loop, new_row_df])

        print(f"Cycle {i_cycle} 建议新实验点 (原始尺度):")
        for k, v_item in new_experiment_data.items():
            if k != output_label:
                print(f"  {k}: {v_item}")
        print(f"  预测 {output_label} (占位符): {new_experiment_data[output_label]:.2f}")

        current_xi = max(min_xi, current_xi * xi_decay_factor) 

    actual_suggestions_made = len(suggested_points_in_this_bo_run_norm)

    if output_label in exp_table_for_bo_loop.columns:
        if exp_table_for_bo_loop[output_label].dtype != 'object':
            exp_table_for_bo_loop[output_label] = exp_table_for_bo_loop[output_label].astype(object)
    else: 
        exp_table_for_bo_loop[output_label] = pd.Series(dtype=object)


    for idx_loop in exp_table_for_bo_loop.index:
        is_bo_suggestion_index = any(str(idx_loop).startswith(f"BO_Cycle{k_loop}_Sugg") for k_loop in range(1, actual_suggestions_made + 1)) 
        if is_bo_suggestion_index: 
            exp_table_for_bo_loop.loc[idx_loop, output_label] = "SUGGESTED_BY_BO_PENDING_RUN"
    
    final_output_df = pd.concat([
        exp_table_full.loc[["MIN", "MAX"]], 
        exp_table_for_bo_loop.drop(["MIN", "MAX"], errors='ignore') 
    ])

    output_filename = f"./GP_BO_Results/BO_Result_CycleEnd.xlsx"
    try:
        final_output_df.to_excel(output_filename)
        print(f"\n[完成] {actual_suggestions_made} 个建议点已生成。总耗时: {time.time()-start_time:.2f}[秒]")
        print(f"最终实验结果（含新建议点）已保存到 {output_filename}")
        
        stopped_early_by_ei = False
        if 'max_ei_value' in locals() and not np.isinf(max_ei_value) and max_ei_value < ei_convergence_threshold and actual_suggestions_made < number_of_experiments_per_cycle and actual_suggestions_made > 0 :
            stopped_early_by_ei = True
            
        if actual_suggestions_made == number_of_experiments_per_cycle and not stopped_early_by_ei:
             print(f"注意: 完成了全部 {number_of_experiments_per_cycle} 次建议，未提前中止。")
        elif actual_suggestions_made < number_of_experiments_per_cycle :
             print(f"注意: 优化在生成 {actual_suggestions_made} 个建议点后提前中止 (可能因为EI过低或CMA-ES未能找到合适点)。")


    except Exception as e:
        print(f"[错误] 保存最终结果到 {output_filename} 失败: {e}")
