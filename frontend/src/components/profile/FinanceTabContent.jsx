import React from 'react';
import { Line, Doughnut } from 'react-chartjs-2';
import BalanceChart from '../charts/BalanceChart';
import AssetsChart from '../charts/AssetsChart';

const FinanceTabContent = ({ 
  user, 
  chartData, 
  chartOptions, 
  chartRef,
  assetsChartData,
  assetsChartOptions
}) => {
  return (
    <div className="lg:col-span-3">
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">Управление капиталом</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-blue-50 p-6 rounded-2xl">
            <div className="text-3xl font-bold text-blue-800 mb-2">${user.balance}</div>
            <div className="text-gray-600">Текущий баланс</div>
            </div>
            
            <div className="bg-green-50 p-6 rounded-2xl">
            <div className="text-3xl font-bold text-green-800 mb-2">+$1,240</div>
            <div className="text-gray-600">Доход за месяц</div>
            </div>
            
            <div className="bg-purple-50 p-6 rounded-2xl">
            <div className="text-3xl font-bold text-purple-800 mb-2">+24.7%</div>
            <div className="text-gray-600">Рост за квартал</div>
            </div>
        </div>
        
        <div className="mb-8">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">Статистика баланса</h3>
            <div className="h-80">
            <Line 
                ref={chartRef}
                data={chartData} 
                options={chartOptions} 
            />
            </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <div>
            <h3 className="text-xl font-semibold text-gray-800 mb-4">Распределение активов</h3>
            <div className="h-64">
                <Doughnut 
                data={assetsChartData} 
                options={assetsChartOptions} 
                />
            </div>
            </div>
            
            <div>
            <h3 className="text-xl font-semibold text-gray-800 mb-4">Прогнозируемый рост</h3>
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-6 rounded-2xl h-full">
                <div className="flex items-center mb-4">
                <div className="w-3 h-3 bg-blue-600 rounded-full mr-2"></div>
                <div className="font-medium">Прогноз на следующий квартал</div>
                </div>
                
                <div className="text-3xl font-bold text-blue-800 mb-2">+18.2%</div>
                <p className="text-gray-600 mb-4">
                На основе текущих рыночных тенденций и стратегии ИИ
                </p>
                
                <div className="flex items-center mb-4">
                <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                <div className="font-medium">Рекомендуемые активы</div>
                </div>
                
                <div className="space-y-2">
                <div className="flex justify-between">
                    <span>Технологические акции</span>
                    <span className="font-medium">+32%</span>
                </div>
                <div className="flex justify-between">
                    <span>Зеленые облигации</span>
                    <span className="font-medium">+18%</span>
                </div>
                <div className="flex justify-between">
                    <span>Криптовалюты</span>
                    <span className="font-medium">+25%</span>
                </div>
                </div>
            </div>
            </div>
        </div>
        
        <div>
            <h3 className="text-xl font-semibold text-gray-800 mb-4">Быстрые действия</h3>
            <div className="flex flex-wrap gap-4">
            <button className="flex-1 min-w-[200px] bg-green-600 hover:bg-green-700 text-white p-4 rounded-lg transition">
                Пополнить счет
            </button>
            <button className="flex-1 min-w-[200px] bg-blue-600 hover:bg-blue-700 text-white p-4 rounded-lg transition">
                Вывести средства
            </button>
            <button className="flex-1 min-w-[200px] bg-purple-600 hover:bg-purple-700 text-white p-4 rounded-lg transition">
                Инвестировать
            </button>
            <button className="flex-1 min-w-[200px] bg-yellow-600 hover:bg-yellow-700 text-white p-4 rounded-lg transition">
                Торговать
            </button>
            </div>
        </div>
        </div>
    </div>
  );
};

export default FinanceTabContent;

