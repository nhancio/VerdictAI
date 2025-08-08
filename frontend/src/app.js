import React from 'react';
import Verdict from './verdict';
import './styles.css';

function App() {
  return (
    <div className="container">
      <h1>VerdictAI</h1>
      <p>Simple AI model to analyse the sentiment of any textual data.</p>
      <Verdict />
    </div>
  );
}

export default App;