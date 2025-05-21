import time
import numpy as np
import torch
import coremltools as ct

# Генерация тестовых данных
def generate_data(size=10_000_000):
    data = np.random.randn(size).astype(np.float32)
    return data

# Метод 1: Обычная обработка на CPU (NumPy)
def numpy_processing(data):
    # Нормализация и FFT
    normalized = (data - np.mean(data)) / np.std(data)
    return np.fft.fft(normalized)

# Метод 2: PyTorch с MPS
def torch_mps_processing(data):
    device = torch.device("mps")
    tensor = torch.tensor(data, device=device)
    
    # Metal-ускоренные операции
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    normalized = (tensor - mean) / std
    fft_result = torch.fft.fft(normalized)
    return fft_result.cpu().numpy()

# # Метод 3: Accelerate Framework
# def accelerate_processing(data):
#     # Использование vDSP и BNNS
#     accelerator = Accelerator()
#     device = accelerator.device
#     print(f"{device=}")
#     # mean = vDSP.mean(data)
#     # std = vDSP.standardDeviation(data)
#     # normalized = vDSP.divide(vDSP.subtract(data, mean), std)
#     # return BNNS.FFT(normalized)

# Метод 4: CoreML
def coreml_processing(data):
    # Создание простой модели для обработки
    model = ct.models.MLModel("preprocessor.mlmodel")
    input_data = ct.TensorType(name="input", shape=data.shape)
    return model.predict({input_data: data})["output"]

# Функция замера времени
def benchmark(func, data, num_runs=10):
    # Прогрев
    _ = func(data[:1000])
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = func(data)
        torch.mps.synchronize()  # Для Metal операций
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return np.mean(times), np.std(times)

if __name__ == "__main__":
    # Генерация данных
    data = generate_data(10_000_000)  # 10 млн точек
    
    # Список методов для сравнения
    methods = [
        ("NumPy CPU", numpy_processing),
        ("PyTorch MPS", torch_mps_processing),
        # ("Accelerate Framework", accelerate_processing),
        ("CoreML", coreml_processing)
    ]
    
    print("Starting benchmarks...")
    print(f"Data size: {data.nbytes / 1e6:.2f} MB")
    
    results = {}
    for name, func in methods:
        try:
            mean_time, std_time = benchmark(func, data)
            results[name] = (mean_time, std_time)
            print(f"{name}: {mean_time:.4f} ± {std_time:.4f} sec")
        except Exception as e:
            print(f"Error in {name}: {str(e)}")
    
    # Вывод результатов
    print("\nResults:")
    for name, (mean, std) in sorted(results.items(), key=lambda x: x[1][0]):
        print(f"{name}:")
        print(f"  Mean time: {mean:.4f} sec")
        print(f"  Std dev:   {std:.4f} sec")
        print(f"  Speedup:   {results['NumPy CPU'][0]/mean:.1f}x")