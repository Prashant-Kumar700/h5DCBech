import subprocess
import matplotlib.pyplot as plt
import os
import csv
import time

def run_benchmark(hdf5_file, dataset_path, num_iterations,params):
    command = f'LD_PRELOAD="/usr/local/hdf5/h5carve.so /usr/local/hdf5/lib/libhdf5.so" ./Benchmark3D {hdf5_file} {dataset_path} {num_iterations} {params}'
    print(command)
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(result.stderr)
    return elapsed_time

def run_benchmark_reexec(hdf5_file, dataset_path, num_iterations,params):
    command = f'LD_PRELOAD="/usr/local/hdf5/h5carve.so /usr/local/hdf5/lib/libhdf5.so" USE_CARVED=1 ./Benchmark3D {hdf5_file} {dataset_path} {num_iterations} {params}'
    print(command)
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(result.stderr)
    return elapsed_time

def main():
    hdf5_file_carved = './LDC2DTEST3D_carved.h5'
    hdf5_file = '~/storage/LDC2DTEST3D.h5'
    dataset_path = '/Timestep_0/id_1'
    # inputs = [(16,16,2)]
    # "Peripheral":[(0,16,16,2),(0,32,32,8),(0,64,64,8),(0,128,128,4),(0,256,256,8),(0,512,512,4)]
    inputs = {"LeftDiagonalEdges":[(1,16,16,2),(1,32,32,8),(1,64,64,8),(1,128,128,4)],
            "RightDiagonalEdges":[(2,16,16,2),(2,32,32,8),(2,64,64,8),(2,128,128,4)],
            "LeftDiagonalStep":[(3,16,16,2,2,2),(3,32,32,8,8,8),(3,64,64,8,8,8),(3,128,128,4,4,4)],
            "Hole":[(4,16,16,2,2,2,8,8),(4,32,32,8,8,8,16,16),(4,64,64,8,8,8,32,32),(4,128,128,4,4,4,64,64)]}


    for i in inputs.keys() :
        print(i)
        for j in inputs[i]:  # Run the benchmark 10 times (adjust as needed)
            params = ' '.join(str(item) for item in j)
            
            time = run_benchmark(hdf5_file, dataset_path, 0,params)
            
            reexectime = run_benchmark_reexec(hdf5_file, dataset_path, 0,params)
            
            compare_and_write_to_csv(i,params,hdf5_file,hdf5_file_carved,'output.csv',time,reexectime)
            os.remove(os.path.expanduser(hdf5_file_carved))
    
    # hdf5_file_carved = './202002_carved.h5'
    # hdf5_file_carved2 = './202002__carved.h5'
    # hdf5_file = '/home/prashant/AISTNotebooks/IMERG-PF/level2/202002.h5'
    # hdf5_file2 = '/home/prashant/AISTNotebooks/GEOS5-PF/level2/202002_.h5'
    # time = runAist()
    # reexectime =runAist_reexec()
    # compare_and_write_to_csv_aist('AIST1','AIST',hdf5_file,hdf5_file_carved,hdf5_file2,hdf5_file_carved2,'output.csv',time,reexectime)
    

def runAist():
    command = f'LD_PRELOAD="/usr/local/hdf5/h5carve.so /usr/local/hdf5/lib/libhdf5.so" python3 ~/AISTNotebooks/POMD-PF.AIST.10202022_h5py.py '
    print(command)
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(result.stderr)
    return elapsed_time

def runAist_reexec():
    command = f'LD_PRELOAD="/usr/local/hdf5/h5carve.so /usr/local/hdf5/lib/libhdf5.so" USE_CARVED=1 python3 ~/AISTNotebooks/POMD-PF.AIST.10202022_h5py.py '
    print(command)
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(result.stderr)
    return elapsed_time

def compare_and_write_to_csv_aist(method,params,file1, file2,file3, file4, output_csv,Time,ReexecTime):
    file1 = os.path.expanduser(file1)
    file2 = os.path.expanduser(file2)
    file3 = os.path.expanduser(file3)
    file4 = os.path.expanduser(file4)
    size1 = os.path.getsize(file1) + os.path.getsize(file3)
    size2 = os.path.getsize(file2) + os.path.getsize(file4)

    reduction_percentage = ((size1 - size2) / size1) * 100

    print(f"Size of {file1}: {size1} bytes")
    print(f"Size of {file2}: {size2} bytes")
    print(f"Reduction in size: {reduction_percentage:.2f}%")

    # Write the results to a CSV file
    with open(output_csv, mode='a', newline='') as csv_file:
        fieldnames = ['Method','Params', 'Original Size', 'Reduced Size', 'Reduction Percentage','Audit Time','Re-Execute Time']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header only if the file is empty
        if os.path.getsize(output_csv) == 0:
            writer.writeheader()

        writer.writerow({'Method': method,'Params':params, 'Original Size': size1, 'Reduced Size': size2, 'Reduction Percentage': reduction_percentage, 'Audit Time': Time, 'Re-Execute Time': ReexecTime})

            
def compare_and_write_to_csv(method,params,file1, file2, output_csv,Time, Reexectime):
    file1 = os.path.expanduser(file1)
    file2 = os.path.expanduser(file2)
    size1 = os.path.getsize(file1)
    size2 = os.path.getsize(file2)

    reduction_percentage = ((size1 - size2) / size1) * 100

    print(f"Size of {file1}: {size1} bytes")
    print(f"Size of {file2}: {size2} bytes")
    print(f"Reduction in size: {reduction_percentage:.2f}%")

    # Write the results to a CSV file
    with open(output_csv, mode='a', newline='') as csv_file:
        fieldnames = ['Method','Params', 'Original Size', 'Reduced Size', 'Reduction Percentage','Audit Time','Re-Execute Time']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header only if the file is empty
        if os.path.getsize(output_csv) == 0:
            writer.writeheader()

        writer.writerow({'Method': method,'Params':params, 'Original Size': size1, 'Reduced Size': size2, 'Reduction Percentage': reduction_percentage, 'Audit Time': Time, 'Re-Execute Time': Reexectime})


if __name__ == "__main__":
    main()