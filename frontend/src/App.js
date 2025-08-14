import { useState, useEffect } from 'react';

const App = () => {
  const [loading, setLoading] = useState(false);
  const [images, setImages] = useState({ before: null, after: null, detection: null });
  const [hasRun, setHasRun] = useState(false);
  const [error, setError] = useState(null);

  // This function simulates the full detection process by calling your backend
  const runDetection = async () => {
    setLoading(true);
    setHasRun(false);
    setError(null);

    try {
      // Make a fetch call to your backend Flask server
      const response = await fetch('http://localhost:5000/api/run-detection');
      const data = await response.json();

      if (response.ok) {
        setImages({
          before: data.beforeImage,
          after: data.afterImage,
          detection: data.detectionImage,
        });
        setHasRun(true);
      } else {
        console.error("Backend error:", data.error);
        setError(`Backend Error: ${data.error}`);
      }
    } catch (error) {
      console.error("Network or server error:", error);
      setError("Network Error: Could not connect to the backend server. Please make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-900 min-h-screen text-gray-100 flex flex-col items-center p-4 md:p-8 font-sans">
      <header className="text-center mb-8">
        <h1 className="text-4xl md:text-5xl font-bold mb-2">Deforestation Detection System</h1>
        <p className="text-md md:text-lg text-gray-400">
          Automated monitoring of the Amazon Rainforest using satellite imagery and a U-Net model.
        </p>
      </header>

      <div className="w-full max-w-4xl flex flex-col items-center">
        <button
          onClick={runDetection}
          disabled={loading}
          className={`
            mb-8 px-8 py-3 rounded-full text-lg font-semibold
            shadow-lg transform transition-all duration-300
            ${loading ? 'bg-gray-600 text-gray-400 cursor-not-allowed animate-pulse' : 'bg-green-600 hover:bg-green-700 text-white hover:scale-105 active:scale-95'}
          `}
        >
          {loading ? 'Analyzing...' : 'Run New Detection'}
        </button>

        {loading && (
          <div className="flex items-center space-x-2 text-xl text-green-500">
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span>Fetching images and running model...</span>
          </div>
        )}

        {error && (
          <div className="bg-red-900 text-red-300 p-4 rounded-lg mb-4 w-full text-center">
            {error}
          </div>
        )}

        {hasRun && (
          <div className="w-full grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
            <div className="bg-gray-800 p-4 rounded-xl shadow-lg">
              <h3 className="text-xl font-bold mb-2 text-center">Before (Pristine)</h3>
              <img src={images.before} alt="Before Deforestation" className="w-full h-auto rounded-lg" />
              <p className="text-sm text-gray-400 mt-2 text-center">Satellite view before deforestation occurred.</p>
            </div>
            <div className="bg-gray-800 p-4 rounded-xl shadow-lg">
              <h3 className="text-xl font-bold mb-2 text-center">After (Deforested)</h3>
              <img src={images.after} alt="After Deforestation" className="w-full h-auto rounded-lg" />
              <p className="text-sm text-gray-400 mt-2 text-center">Latest satellite view with deforestation.</p>
            </div>
            <div className="bg-gray-800 p-4 rounded-xl shadow-lg">
              <h3 className="text-xl font-bold mb-2 text-center">Detected Deforestation</h3>
              <img src={images.detection} alt="Detected Deforestation" className="w-full h-auto rounded-lg" />
              <p className="text-sm text-red-400 mt-2 text-center">Model's output showing cleared areas in red.</p>
            </div>
          </div>
        )}

        {hasRun && (
          <div className="bg-gray-800 p-6 rounded-xl shadow-lg mt-8 w-full max-w-xl text-center">
            <p className="text-lg font-bold text-red-400">DEFORESTATION DETECTED!</p>
            <p className="text-gray-300 mt-2">The model has successfully identified significant deforestation in the latest satellite image. Further analysis may be required.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
