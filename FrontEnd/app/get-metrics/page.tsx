'use client';

import React, { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import axios from 'axios';
import toast, { Toaster } from 'react-hot-toast';
import ReactLoading from "react-loading";
import MUIDataTable, { MUIDataTableOptions } from "mui-datatables";
import { createTheme, ThemeProvider } from '@mui/material/styles';



const MetricPage: React.FC = () => {
  const searchParams = useSearchParams();
  const formData = {
    model_name: searchParams.get('huggingfaceModel') || '',
    dataset_name: searchParams.get('huggingfaceDataset') || '',
    huggingface_token: searchParams.get('huggingfaceToken') || '',
    wandb_token: searchParams.get('wandbToken') || '',
    model_subset: searchParams.get('modelSubset') || ''
  };
  const [loading, setLoading] = useState<boolean>(false);
  const [baseResults, setBaseResults] = useState<any>(null);
  const [optimResults, setOptimResults] = useState<any>(null);
  const [zeroResults, setZeroResults] = useState<any>(null);
  const [xtcResults, setXtcResults] = useState<any>(null);
  const [weightResults, setWeightResults] = useState<any>(null);
  const [pruningResults, setPruningResults] = useState<any>(null);
  const [ptqResults, setPtqResults] = useState<any>(null);
  const [w8a8Results, setW8a8Results] = useState<any>(null);

  const [baseid, setBaseid] = useState<string>('');
  const [optimid, setOptimid] = useState<string>('');

  const [loadingMessage, setLoadingMessage] = useState<String>('');

  const data = useMemo(() => {
    if (!baseResults || !optimResults) return [];

    const formatData = (result: any, modelName: any) => ({
      model: modelName,
      eval_loss: result.eval_result.eval_loss,
      eval_accuracy: result.eval_result.eval_accuracy,
      eval_runtime: result.eval_result.eval_runtime,
      eval_samples_per_second: result.eval_result.eval_samples_per_second,
      eval_steps_per_second: result.eval_result.eval_steps_per_second,
      epoch: result.eval_result.epoch,
      avg_cpu_utilization: result.avg_cpu_utilization,
      avg_memory_utilization: result.avg_memory_utilization,
      avg_gpu_utilization: result.avg_gpu_utilization,
      total_time: result.total_time
    });

    return [
      formatData(baseResults, 'Base Model'),
      formatData(optimResults, 'Optimized Model'),
      formatData(zeroResults, 'ZeroQuant Model'),
      formatData(xtcResults, 'Extreme Compression Model'),
      formatData(weightResults, 'Weight Quantization Model'),
      formatData(pruningResults, 'Unstructured Pruned Model'),
      formatData(ptqResults, 'Post Training Quantization Model'),
      formatData(w8a8Results, 'W8A8 Model')
    ];
  }, [baseResults, optimResults]);


  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        setLoadingMessage('Loading the Input Model');
        const baseResponse = await axios.post('http://127.0.0.1:5000/api/base', formData);
        setBaseResults(baseResponse.data.eval_results);
        setBaseid(baseResponse.data.run_id);

        setLoadingMessage('Applying Zero Quantization');
        const optimResponse1 = await axios.post('http://127.0.0.1:5000/api/zeroquant', formData);
        setZeroResults(optimResponse1.data.eval_results);
        setOptimid(optimResponse1.data.run_id);

        setLoadingMessage('Applying Extreme Compression');
        const optimResponse2 = await axios.post('http://127.0.0.1:5000/api/XTC', formData);
        setXtcResults(optimResponse2.data.eval_results);
        setOptimid(optimResponse2.data.run_id);

        setLoadingMessage('Applying Weight Quantization');
        const optimResponse3 = await axios.post('http://127.0.0.1:5000/api/weight_quant', formData);
        setWeightResults(optimResponse3.data.eval_results);
        setOptimid(optimResponse3.data.run_id);
        
        setLoadingMessage('Applying Post Training Quantization');
        const optimResponse4 = await axios.post('http://127.0.0.1:5000/api/PTQuant', formData);
        setPtqResults(optimResponse4.data.eval_results);
        setOptimid(optimResponse4.data.run_id);

        setLoadingMessage('Applying W8A8 Quantization');
        const optimResponse5 = await axios.post('http://127.0.0.1:5000/api/w8a8', formData);
        setW8a8Results(optimResponse5.data.eval_results);
        setOptimid(optimResponse5.data.run_id);

        setLoadingMessage('Applying Unstructured Pruning');
        const optimResponse6 = await axios.post('http://127.0.0.1:5000/api/pruning', formData);
        setPruningResults(optimResponse6.data.eval_results);
        setOptimid(optimResponse6.data.run_id);

        setLoadingMessage('Applying optimisation techniques');
        const optimResponse = await axios.post('http://127.0.0.1:5000/api/optimise', formData);
        setOptimResults(optimResponse.data.eval_results);
        setOptimid(optimResponse.data.run_id);

        setLoadingMessage('Evaluating the model');

        toast.success('Data fetched successfully!')

      } catch (error) {
        console.log('Error fetching data:', error)
        toast.error('Error fetching data!')
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const columns = [
    { name: 'model', label: 'Model' },
    { name: 'eval_loss', label: 'Eval Loss' },
    { name: 'eval_accuracy', label: 'Eval Accuracy' },
    { name: 'eval_runtime', label: 'Runtime Latency' },
    { name: 'eval_samples_per_second', label: 'Throughput' },
    { name: 'eval_steps_per_second', label: 'Steps/Sec' }
  ];

  const columns2 = [
    { name: 'model', label: 'Model' },
    { name: 'avg_cpu_utilization', label: 'CPU Utilization(%)' },
    { name: 'avg_memory_utilization', label: 'Memory Utilization(%)' },
    { name: 'avg_gpu_utilization', label: 'GPU Utilization(%)' },
    { name: 'total_time', label: 'Total Latency' }
  ];

  const options: MUIDataTableOptions = {
    selectableRows: 'none',
    filterType: 'checkbox',
    responsive: 'standard',
    rowsPerPage: 5,
    rowsPerPageOptions: [5, 10, 15],
    elevation: 0
  };

  const getMuiTheme = () => createTheme({
    components: {
      MuiTable: {
        styleOverrides: {
          root: {
            backgroundColor: '#000000',
            color: '#ffffff'
          }
        }
      },
      MuiToolbar: {
        styleOverrides: {
          root: {
            backgroundColor: '#000000',
            color: '#ffffff'
          }
        }
      },
      MuiTableCell: {
        styleOverrides: {
          root: {
            borderColor: '#808080',
            color: '#ffffff'
          },
          head: {
            backgroundColor: '#000000',
            color: '#ffffff'
          },
          footer:{
            backgroundColor: '#000000',
            color: '#000000',
            visibility: 'hidden'
          }
        }
      },
      MuiTableHead: {
        styleOverrides: {
          root: {
            backgroundColor: '#000000',
          }
        }
      },
      MuiTypography: {
        styleOverrides: {
          root: {
            color: '#ffffff'
          }
        }
      },
    }
  });




  return (
    <>
      <Toaster />
      {loading && (
        <div className='App'>
          <ReactLoading
            type={"bars"}
            color={"#ffffff"}
            height={100}
            width={100}
          />
          <p className="loadingText">{loadingMessage}</p>
          <style jsx>{`
        .App {
          width: 100vw;
          height: 100vh;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
        }
      `}</style>
        </div>

      )}
      <ThemeProvider theme={getMuiTheme()}>
        {!loading && baseResults && optimResults && (
          <MUIDataTable
            title={"Evaluation Metrics Comparison"}
            data={data}
            columns={columns}
            options={options}
          />
        )}
        <br></br>
        <br></br>
        {!loading && baseResults && optimResults && (
          <MUIDataTable
            title={"Resource Utilization Comparison"}
            data={data}
            columns={columns2}
            options={options}
          />
        )}
      </ThemeProvider>
      {!loading && baseResults && optimResults && (
        <div className='flex items-center justify-center m-4 mx-auto'>
          <button className='btn btn-outline btn-wide gradient-button'>
            <a href={`http://127.0.0.1:5000/api/download/OptimisedModel.zip`} download>Download Optimised Model</a>
          </button>
          <style jsx>{`
        .gradient-button {
          border: 2px solid;
          border-image-slice: 2;
          border-width: 2px;
          border-image-source: linear-gradient(to right, #9b5afc, #00CC99);
          background: transparent;
          color: #9b5afc;
          padding: 10px 20px;
          font-size: 1rem;
          font-weight: bold;
          cursor: pointer;
          transition: all 0.3s ease;
        }
        
        .gradient-button:hover {
          background: linear-gradient(to right, #9b5afc, #00CC99);
          color: white;
        }
      `}</style>
        </div>
      )}
    </>
  );
};

export default MetricPage;
