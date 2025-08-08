import React, { useState } from 'react';

function Verdict() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);

  const handleAnalyze = async () => {
    const response = await fetch('http://localhost:5000/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    const data = await response.json();
    setResult(data);
  };

  return (
    <div>
      <textarea
        rows="6"
        placeholder="Enter text to analyze..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button onClick={handleAnalyze}>Analyze</button>

      {result && (
        <div className="result">
          <h2>Major Emotion: {result.Predicted_Sentiment} - {result.cd}%</h2>
          <ul>
            {Object.entries(result.emotions).map(([emotion, prob]) => (
              <li key={emotion}>
                {emotion}: {prob}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default Verdict;