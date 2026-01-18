import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Dashboard from './pages/Dashboard';
import Signals from './pages/Signals';
import Analytics from './pages/Analytics';
import Portfolio from './pages/Portfolio';
import Navigation from './components/Navigation';
import './App.css';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-gray-900 text-white">
          <Navigation />
          <main className="container mx-auto px-4 py-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/signals" element={<Signals />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/portfolio" element={<Portfolio />} />
            </Routes>
          </main>
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
