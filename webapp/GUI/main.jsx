import React, { useEffect, useState } from 'react';
import ReactDOM from 'react-dom/client';
import AudioSimilarityPage from './hmong_audio_similarity_ui_react.jsx';
import DataCollectionPage from './DataCollectionPage.jsx';
import BatchScorePage from './BatchScorePage.jsx';
import './index.css';

// Landing page when there is no hash. Static (collection-only) builds set
// VITE_DEFAULT_PAGE=collect so the site opens on the recorder; the full
// container build leaves it on the eval tool.
const DEFAULT_PAGE = (import.meta.env.VITE_DEFAULT_PAGE === 'collect') ? 'collect' : 'eval';

const PAGES = ['eval', 'batch', 'collect'];
const routeFromHash = () => {
  const h = window.location.hash.replace(/^#\/?/, '');
  return PAGES.includes(h) ? h : DEFAULT_PAGE;
};

function App() {
  const [page, setPage] = useState(routeFromHash);
  useEffect(() => {
    const onHash = () => setPage(routeFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  const navigate = (r) => { window.location.hash = `#/${r}`; };
  if (page === 'collect') return <DataCollectionPage page={page} onNavigate={navigate} />;
  if (page === 'batch') return <BatchScorePage page={page} onNavigate={navigate} />;
  return <AudioSimilarityPage page={page} onNavigate={navigate} />;
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
