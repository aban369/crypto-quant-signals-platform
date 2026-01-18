import React, { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { TrendingUp, TrendingDown, Activity, AlertTriangle } from 'lucide-react';
import SignalCard from '../components/SignalCard';
import EconophysicsChart from '../components/EconophysicsChart';
import OrderBookHeatmap from '../components/OrderBookHeatmap';
import { fetchAllSignals, fetchEconophysics } from '../services/api';

const Dashboard: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT');

  const { data: signals, isLoading: signalsLoading } = useQuery({
    queryKey: ['signals'],
    queryFn: fetchAllSignals,
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  const { data: econophysics } = useQuery({
    queryKey: ['econophysics', selectedSymbol],
    queryFn: () => fetchEconophysics(selectedSymbol),
    refetchInterval: 5000,
  });

  const symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">
            Crypto Quant Signals Platform
          </h1>
          <p className="text-gray-400 mt-2">
            Advanced analysis powered by 17+ research papers
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Activity className="w-5 h-5 text-green-500 animate-pulse" />
          <span className="text-sm text-gray-400">Live</span>
        </div>
      </div>

      {/* Symbol Selector */}
      <div className="flex space-x-4">
        {symbols.map((symbol) => (
          <button
            key={symbol}
            onClick={() => setSelectedSymbol(symbol)}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              selectedSymbol === symbol
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/50'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {symbol}
          </button>
        ))}
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <MetricCard
          title="Temperature"
          value={econophysics?.temperature?.toFixed(2) || '0.00'}
          icon={<TrendingUp className="w-6 h-6" />}
          color="orange"
          subtitle={econophysics?.phase || 'Loading...'}
        />
        <MetricCard
          title="Entropy"
          value={econophysics?.entropy?.toFixed(2) || '0.00'}
          icon={<Activity className="w-6 h-6" />}
          color="purple"
          subtitle="Market Disorder"
        />
        <MetricCard
          title="Pressure"
          value={econophysics?.pressure?.toFixed(2) || '0.00'}
          icon={econophysics?.pressure > 0 ? <TrendingUp className="w-6 h-6" /> : <TrendingDown className="w-6 h-6" />}
          color={econophysics?.pressure > 0 ? 'green' : 'red'}
          subtitle={econophysics?.pressure > 0 ? 'Buy Pressure' : 'Sell Pressure'}
        />
        <MetricCard
          title="Flash Crash Risk"
          value="Low"
          icon={<AlertTriangle className="w-6 h-6" />}
          color="yellow"
          subtitle="Hawkes Process"
        />
      </div>

      {/* Signals Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {signalsLoading ? (
          <div className="col-span-3 text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
            <p className="text-gray-400 mt-4">Loading signals...</p>
          </div>
        ) : (
          symbols.map((symbol) => (
            <SignalCard
              key={symbol}
              symbol={symbol}
              signal={signals?.[symbol]}
            />
          ))
        )}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <EconophysicsChart symbol={selectedSymbol} />
        <OrderBookHeatmap symbol={selectedSymbol} />
      </div>

      {/* Research Papers Info */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-xl font-bold mb-4">Implemented Research</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <h4 className="font-semibold text-blue-400 mb-2">Econophysics</h4>
            <ul className="space-y-1 text-gray-400">
              <li>• Thermodynamic Analysis</li>
              <li>• Statistical Physics</li>
              <li>• Power-Law Distributions</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-purple-400 mb-2">Market Microstructure</h4>
            <ul className="space-y-1 text-gray-400">
              <li>• Order Flow Imbalance</li>
              <li>• DeepLOB CNN</li>
              <li>• Limit Order Book Dynamics</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-green-400 mb-2">Advanced Models</h4>
            <ul className="space-y-1 text-gray-400">
              <li>• Hawkes Process</li>
              <li>• RL Trading Agents</li>
              <li>• Portfolio Optimization</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

interface MetricCardProps {
  title: string;
  value: string;
  icon: React.ReactNode;
  color: string;
  subtitle: string;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, icon, color, subtitle }) => {
  const colorClasses = {
    orange: 'from-orange-500 to-red-500',
    purple: 'from-purple-500 to-pink-500',
    green: 'from-green-500 to-emerald-500',
    red: 'from-red-500 to-rose-500',
    yellow: 'from-yellow-500 to-orange-500',
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 hover:border-gray-600 transition-all">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-gray-400 text-sm font-medium">{title}</h3>
        <div className={`bg-gradient-to-r ${colorClasses[color]} p-2 rounded-lg`}>
          {icon}
        </div>
      </div>
      <div className="text-3xl font-bold mb-1">{value}</div>
      <div className="text-sm text-gray-500">{subtitle}</div>
    </div>
  );
};

export default Dashboard;
