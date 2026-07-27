import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cached, setCached] = useState(null);
  const [latency, setLatency] = useState(null);
  const [sessionId] = useState('s_' + Date.now());
  const [history, setHistory] = useState([]);
  const [answerVisible, setAnswerVisible] = useState(false);
  const answerRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    if (answer) {
      setAnswerVisible(false);
      requestAnimationFrame(() => {
        setTimeout(() => setAnswerVisible(true), 30);
      });
      setTimeout(() => {
        if (answerRef.current) {
          answerRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
    }
  }, [answer]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);

  const handleSubmit = async () => {
    if (!question.trim() || loading) return;
    setLoading(true);
    setError(null);
    setAnswer(null);
    setSources([]);
    setCached(null);
    setLatency(null);
    setAnswerVisible(false);
    try {
      const response = await axios.post(API_BASE + '/query', {
        question: question.trim(),
        session_id: sessionId
      });
      const data = response.data;
      setAnswer(data.answer);
      setSources(data.sources || []);
      setCached(data.cached);
      setLatency(data.latency_ms);
      setHistory(function(prev) {
        return prev.concat([{
          question: question.trim(),
          answer: data.answer,
          cached: data.cached,
          latency: data.latency_ms
        }]);
      });
      setQuestion('');
    } catch (err) {
      if (err.response) {
        setError(err.response.data.detail || 'Something went wrong.');
      } else {
        setError('Cannot reach the backend. Make sure uvicorn is running on port 8000.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleClear = async () => {
    try {
      await axios.delete(API_BASE + '/session/' + sessionId);
    } catch (e) {}
    setAnswer(null);
    setSources([]);
    setError(null);
    setCached(null);
    setLatency(null);
    setHistory([]);
    setQuestion('');
    setAnswerVisible(false);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  const formatLatency = function(ms) {
    if (ms === null) return '';
    if (ms < 1000) return ms.toFixed(1) + 'ms';
    return (ms / 1000).toFixed(2) + 's';
  };

  const examples = [
    "What is LoRA and how does it reduce parameters?",
    "How does BERT use masked language modeling?",
    "What is dense passage retrieval?",
    "How does RAG combine retrieval with generation?"
  ];

  const showEmpty = !answer && !loading && !error && history.length === 0;
  const historyToShow = history.slice(0, answer ? -1 : undefined);

  return (
    <div className="app">
      <div className="bg-glow"></div>

      <header className="header">
        <div className="header-eyebrow">RAG - 1,000+ ArXiv Papers</div>
        <h1 className="header-title">ML Research Assistant</h1>
        <p className="header-sub">
          Ask anything about machine learning research.
          Answers grounded in real papers, with citations.
        </p>
      </header>

      <main className="main">

        {showEmpty && (
          <div className="empty-state">
            <p className="empty-label">Try asking</p>
            <div className="examples">
              {examples.map(function(q, i) {
                return (
                  <button
                    key={i}
                    className="example-btn"
                    onClick={function() {
                      setQuestion(q);
                      if (textareaRef.current) textareaRef.current.focus();
                    }}
                  >
                    {q}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {historyToShow.length > 0 && (
          <div className="history">
            {historyToShow.map(function(item, i) {
              return (
                <div key={i} className="history-item">
                  <div className="history-q">
                    <span className="history-q-label">Q</span>
                    {item.question}
                  </div>
                  <div className="history-a">
                    {item.answer.length > 200 ? item.answer.slice(0, 200) + '...' : item.answer}
                  </div>
                  <div className="history-meta">
                    <span className={item.cached ? 'badge badge-cached' : 'badge badge-fresh'}>
                      {item.cached ? 'cached' : 'retrieved'}
                    </span>
                    <span className="history-latency">{formatLatency(item.latency)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {loading && (
          <div className="loading-state">
            <div className="loading-bar">
              <div className="loading-bar-fill"></div>
            </div>
            <p className="loading-text">Searching 1,000+ papers</p>
          </div>
        )}

        {error && (
          <div className="error-state">
            <span className="error-icon">!</span>
            <p>{error}</p>
          </div>
        )}

        {answer && (
          <div ref={answerRef} className={answerVisible ? 'answer-card answer-visible' : 'answer-card'}>
            <div className="answer-meta">
              <span className={cached ? 'badge badge-cached' : 'badge badge-fresh'}>
                {cached ? 'cached' : 'retrieved'}
              </span>
              <span className="answer-latency">{formatLatency(latency)}</span>
              <span className="answer-session">session {sessionId.slice(-6)}</span>
            </div>
            <div className="answer-question">
              {history.length > 0 ? history[history.length - 1].question : ''}
            </div>
            <div className="answer-body">
              <div className="answer-accent-bar"></div>
              <p className="answer-text">{answer}</p>
            </div>
            {sources.length > 0 && (
              <div className="sources">
                <div className="sources-label">
                  Sources - {sources.length} {sources.length > 1 ? 'papers' : 'paper'}
                </div>
                <div className="sources-list">
                  {sources.map(function(s, i) {
                    return (
                      <div key={i} className="source-row">
                        <span className="source-index">
                          {i < 9 ? '0' + (i + 1) : '' + (i + 1)}
                        </span>
                        <div className="source-info">
                          <a
                            className="source-title"
                            href={'https://arxiv.org/search/?searchtype=all&query=' + encodeURIComponent(s.title)}
                            target="_blank"
                            rel="noreferrer"
                          >
                            {s.title}
                          </a>
                          <span className="source-detail">
                            {s.year} - relevance {(s.relevance_score || 0).toFixed(3)}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

      </main>

      <div className="input-dock">
        <div className="input-inner">
          <textarea
            ref={textareaRef}
            className="input-field"
            value={question}
            onChange={function(e) { setQuestion(e.target.value); }}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about ML research..."
            rows={2}
            disabled={loading}
          />
          <div className="input-actions">
            <span className="char-count">{question.length > 0 ? question.length + ' chars' : ''}</span>
            <button className="btn-clear" onClick={handleClear} disabled={loading}>
              New chat
            </button>
            <button
              className="btn-submit"
              onClick={handleSubmit}
              disabled={loading || !question.trim()}
            >
              {loading ? <span className="btn-spinner"></span> : 'Ask'}
            </button>
          </div>
        </div>
        <p className="input-hint">Shift+Enter for new line - Enter to submit</p>
      </div>

    </div>
  );
}

export default App;
