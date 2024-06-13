'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';

const FormPage: React.FC = () => {
  const router = useRouter();
  const [formData, setFormData] = useState({
    huggingfaceModel: '',
    huggingfaceDataset: '',
    huggingfaceToken: '',
    wandbToken: '',
    modelSubset: ''
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const queryString = new URLSearchParams(formData).toString();
    router.push(`/get-metrics?${queryString}`);
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-black">
      <div className="w-full max-w-3xl p-8 bg-black border border-spacing-4 rounded-2xl shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-center">Model Configuration</h2>
        <form className="space-y-6" onSubmit={handleSubmit}>
          <div className="flex space-x-6">
            <div className="form-control w-1/2">
              <label className="label">
                <span className="label-text">HuggingFace Model</span>
              </label>
              <input
                type="text"
                name="huggingfaceModel"
                value={formData.huggingfaceModel}
                onChange={handleChange}
                placeholder="Enter HuggingFace Model"
                className="input input-bordered w-full"
              />
            </div>
            <div className="form-control w-1/2">
              <label className="label">
                <span className="label-text">HuggingFace Dataset</span>
              </label>
              <input
                type="text"
                name="huggingfaceDataset"
                value={formData.huggingfaceDataset}
                onChange={handleChange}
                placeholder="Enter HuggingFace Dataset"
                className="input input-bordered w-full"
              />
            </div>
          </div>
          <div className="flex space-x-6">
            <div className="form-control w-1/2">
              <label className="label">
                <span className="label-text">HuggingFace Token</span>
              </label>
              <input
                type="text"
                name="huggingfaceToken"
                value={formData.huggingfaceToken}
                onChange={handleChange}
                placeholder="Enter HuggingFace Token"
                className="input input-bordered w-full"
              />
            </div>
            <div className="form-control w-1/2">
              <label className="label">
                <span className="label-text">WandB Token</span>
              </label>
              <input
                type="text"
                name="wandbToken"
                value={formData.wandbToken}
                onChange={handleChange}
                placeholder="Enter WandB Token"
                className="input input-bordered w-full"
              />
            </div>
          </div>
          <div className="form-control">
            <label className="label">
              <span className="label-text">Model Subset</span>
            </label>
            <select
              name="modelSubset"
              value={formData.modelSubset}
              onChange={handleChange}
              className="select select-bordered w-full"
            >
              <option value="None">None</option>
              <option value="sequence-classification">Sequence Classification</option>
              <option value="vision-transformer">Vision Transformer</option>
              <option value="automodel">AutoModel</option>
            </select>
          </div>
          <button type="submit" className="btn gradient-button w-full mt-4">Submit</button>
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
        </form>
      </div>
    </div>
  );
};

export default FormPage;
