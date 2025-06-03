import React, { useState, useRef, useEffect } from 'react';
import logo from '../img/logo.webp'; 
import { Line, Doughnut } from 'react-chartjs-2';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement, 
  Title, 
  Tooltip, 
  Legend,
  ArcElement,
  Filler
} from 'chart.js';

// Регистрируем компоненты Chart.js
ChartJS.register(
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement, 
  Title, 
  Tooltip, 
  Legend,
  ArcElement,
  Filler
);

const ProfilePage = ({ user, onLogout }) => {
  const [activeTab, setActiveTab] = useState('profile');
  const [assetsData, setAssetsData] = useState(null);
  const chartRef = useRef(null);
  
  // Генерация данных для графика
  useEffect(() => {
    // Данные для графика баланса
    const labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    
    // Фактические данные
    const actualData = Array(6).fill(null)
      .map((_, i) => user.balance - 1000 + i * 500)
      .concat(Array(6).fill(null));
    
    // Прогнозируемые данные
    const forecastData = Array(6).fill(null)
      .concat(Array(6).fill(null)
      .map((_, i) => user.balance + (i + 1) * 300));
    
    // Данные для распределения активов
    setAssetsData({
      stocks: Math.floor(Math.random() * 50) + 30,
      crypto: Math.floor(Math.random() * 30) + 15,
      bonds: Math.floor(Math.random() * 20) + 10,
      commodities: Math.floor(Math.random() * 10) + 5,
      cash: Math.floor(Math.random() * 10) + 5,
    });
    
    // Настройка градиента для графика
    if (chartRef.current) {
      const ctx = chartRef.current.ctx;
      const gradient = ctx.createLinearGradient(0, 0, 0, 400);
      gradient.addColorStop(0, 'rgba(79, 70, 229, 0.6)');
      gradient.addColorStop(1, 'rgba(79, 70, 229, 0.05)');
      
      setChartData({
        labels,
        datasets: [
          {
            label: 'Actual Balance',
            data: actualData,
            borderColor: 'rgb(79, 70, 229)',
            backgroundColor: gradient,
            pointBackgroundColor: 'rgb(79, 70, 229)',
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: 'rgb(79, 70, 229)',
            tension: 0.4,
            fill: true,
          },
          {
            label: 'Forecast',
            data: forecastData,
            borderColor: 'rgb(14, 165, 233)',
            backgroundColor: 'rgba(14, 165, 233, 0.05)',
            borderDash: [5, 5],
            pointBackgroundColor: 'rgb(14, 165, 233)',
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: 'rgb(14, 165, 233)',
            tension: 0.4,
            fill: true,
          }
        ]
      });
    }
  }, [user.balance]);

  const [chartData, setChartData] = useState({
    labels: [],
    datasets: []
  });

  // Настройки графика баланса
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 12,
        titleFont: {
          size: 14,
          weight: 'bold'
        },
        bodyFont: {
          size: 13
        },
        callbacks: {
          label: function(context) {
            const datasetLabel = context.dataset.label || '';
            const value = context.parsed.y || 0;
            let label = `${datasetLabel}: $${value.toFixed(2)}`;
            
            if (datasetLabel === 'Forecast') {
              const change = ((value - user.balance) / user.balance * 100).toFixed(2);
              label += ` (${change}%)`;
            }
            
            return label;
          },
          afterLabel: function(context) {
            if (context.datasetIndex === 1) {
              return 'Projected growth based on current trends';
            }
            return null;
          }
        }
      }
    },
    scales: {
      x: {
        grid: {
          display: false
        }
      },
      y: {
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        },
        ticks: {
          callback: function(value) {
            return '$' + value;
          }
        }
      }
    },
    interaction: {
      mode: 'index',
      intersect: false,
    },
    hover: {
      mode: 'index',
      intersect: false
    }
  };

  // Данные для круговой диаграммы активов
  const assetsChartData = {
    labels: ['Stocks', 'Crypto', 'Bonds', 'Commodities', 'Cash'],
    datasets: [
      {
        data: assetsData ? [
          assetsData.stocks,
          assetsData.crypto,
          assetsData.bonds,
          assetsData.commodities,
          assetsData.cash
        ] : [30, 25, 20, 15, 10],
        backgroundColor: [
          'rgba(79, 70, 229, 0.8)',
          'rgba(14, 165, 233, 0.8)',
          'rgba(34, 197, 94, 0.8)',
          'rgba(234, 179, 8, 0.8)',
          'rgba(107, 114, 128, 0.8)'
        ],
        borderColor: [
          'rgba(79, 70, 229, 1)',
          'rgba(14, 165, 233, 1)',
          'rgba(34, 197, 94, 1)',
          'rgba(234, 179, 8, 1)',
          'rgba(107, 114, 128, 1)'
        ],
        borderWidth: 1,
      }
    ]
  };

  const assetsChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
        labels: {
          boxWidth: 12,
          padding: 16,
          font: {
            size: 12
          }
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.label || '';
            const value = context.parsed || 0;
            const percentage = Math.round(value);
            return `${label}: ${percentage}%`;
          }
        }
      }
    },
    cutout: '70%',
  };

  // Данные для финансовой активности
  const financialActivity = [
    { type: 'EXPENSE', amount: 1240, description: 'Market fees', date: 'TODAY, 15:30' },
    { type: 'INCOME', amount: 3200, description: 'Dividend payment', date: 'TODAY, 12:45' },
    { type: 'TRANSFER', amount: 5000, description: 'Account funding', date: 'YESTERDAY, 09:22' },
    { type: 'INCOME', amount: 1870, description: 'Stock sale', date: 'JUN 02, 16:15' },
  ];

  return (
    <div className="min-h-screen bg-gray-100 flex">
      {/* Боковая панель */}
      <div className="w-64 bg-gradient-to-b from-blue-900 to-purple-900 text-white p-6 flex flex-col">
        <div className="mb-10">
            <img 
                src={logo} 
                alt="AI Trading Logo" 
                className="w-20 h-auto"
            />
            <h1 className="text-2xl font-bold">Agent Trade</h1>
        </div>
        
         <nav className="flex-1">
          <ul className="space-y-4">
            <li>
              <button 
                className={`w-full text-left p-3 rounded-lg transition ${activeTab === 'profile' ? 'bg-white text-blue-900' : 'hover:bg-blue-800'}`}
                onClick={() => setActiveTab('profile')}
              >
                <div className="flex items-center">
                  <span className="ml-2">Профиль</span>
                </div>
              </button>
            </li>
            <li>
              <button 
                className={`w-full text-left p-3 rounded-lg transition ${activeTab === 'finance' ? 'bg-white text-blue-900' : 'hover:bg-blue-800'}`}
                onClick={() => setActiveTab('finance')}
              >
                <div className="flex items-center">
                  <span className="ml-2">Финансы</span>
                </div>
              </button>
            </li>
            <li>
              <button className="w-full text-left p-3 rounded-lg hover:bg-blue-800 transition">
                <div className="flex items-center">
                  <span className="ml-2">Транзакции</span>
                </div>
              </button>
            </li>
            <li>
              <button className="w-full text-left p-3 rounded-lg hover:bg-blue-800 transition">
                <div className="flex items-center">
                  <span className="ml-2">История операций</span>
                </div>
              </button>
            </li>
            <li>
              <button className="w-full text-left p-3 rounded-lg hover:bg-blue-800 transition">
                <div className="flex items-center">
                  <span className="ml-2">Настройки</span>
                </div>
              </button>
            </li>
          </ul>
        </nav>
        
        <div className="mt-auto">
          <button 
            className="w-full bg-red-600 hover:bg-red-700 text-white p-3 rounded-lg transition"
            onClick={onLogout}
          >
            Выйти
          </button>
        </div>
      </div>
      
      {/* Основное содержимое */}
      <div className="flex-1 p-8 overflow-auto">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Блок с приветствием */}
          <div className="lg:col-span-3 bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Добро пожаловать, {user.login}!</h2>
            <p className="text-gray-600">
              Наша ИИ система анализирует рынок 24/7 и совершает сделки с максимальной эффективностью.
            </p>
          </div>
          
          {/* Контент в зависимости от активной вкладки */}
          {activeTab === 'profile' && (
            <div className="lg:col-span-3">
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h2 className="text-2xl font-bold text-gray-800 mb-6">Личная информация</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-700 mb-4">Основные данные</h3>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm text-gray-500 mb-1">Имя пользователя</label>
                        <div className="p-3 bg-gray-50 rounded-lg">{user.login}</div>
                      </div>
                      <div>
                        <label className="block text-sm text-gray-500 mb-1">Email</label>
                        <div className="p-3 bg-gray-50 rounded-lg">{user.email}</div>
                      </div>
                      <div>
                        <label className="block text-sm text-gray-500 mb-1">Дата регистрации</label>
                        <div className="p-3 bg-gray-50 rounded-lg">01 января 2025</div>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold text-gray-700 mb-4">Безопасность</h3>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm text-gray-500 mb-1">Статус аккаунта</label>
                        <div className="p-3 bg-green-50 text-green-700 rounded-lg">Подтвержден</div>
                      </div>
                      <button className="w-full bg-blue-100 text-blue-700 hover:bg-blue-200 p-3 rounded-lg transition">
                        Сменить пароль
                      </button>
                      <button className="w-full bg-blue-100 text-blue-700 hover:bg-blue-200 p-3 rounded-lg transition">
                        Двухфакторная аутентификация
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {activeTab === 'finance' && (
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
          )}
          
          {/* Блок с финансовой активностью */}
          <div className="lg:col-span-2 bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Finance Activity</h2>
            <div className="space-y-4">
              {financialActivity.map((activity, index) => (
                <div 
                  key={index} 
                  className={`p-4 rounded-lg ${
                    activity.type === 'EXPENSE' ? 'bg-red-50' : 
                    activity.type === 'INCOME' ? 'bg-green-50' : 
                    'bg-blue-50'
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <div className="font-semibold">{activity.description}</div>
                    <div className={`font-bold ${
                      activity.type === 'EXPENSE' ? 'text-red-700' : 
                      activity.type === 'INCOME' ? 'text-green-700' : 
                      'text-blue-700'
                    }`}>
                      {activity.type === 'EXPENSE' ? '-' : '+'}${activity.amount}
                    </div>
                  </div>
                  <div className="flex justify-between mt-2 text-sm text-gray-500">
                    <div className="flex items-center">
                      <div className={`w-3 h-3 rounded-full mr-2 ${
                        activity.type === 'EXPENSE' ? 'bg-red-500' : 
                        activity.type === 'INCOME' ? 'bg-green-500' : 
                        'bg-blue-500'
                      }`}></div>
                      <span>{activity.type}</span>
                    </div>
                    <span>{activity.date}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Блок с чатами */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Chats</h2>
            <div className="space-y-4">
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="font-semibold">Support Team</div>
                <p className="text-sm text-gray-600 mt-1">
                  I hope you are finding Blocs fun to build websites with. Thanks for your support!
                </p>
                <div className="text-xs text-gray-400 mt-2">TODAY, 14:30</div>
              </div>
              <div className="p-4 bg-blue-50 rounded-lg">
                <div className="font-semibold">Norm</div>
                <p className="text-sm text-gray-600 mt-1">
                  Hey Norm, Blocs Rocks, congrats!
                </p>
                <div className="text-xs text-gray-400 mt-2">TODAY, 19:30</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;