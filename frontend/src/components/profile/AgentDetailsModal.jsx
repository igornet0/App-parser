import React from 'react';

const AgentDetailsModal = ({ agent, onClose }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-2xl max-h-[90vh] overflow-auto">
        <div className="p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-bold text-gray-800">Детали агента: {agent.name}</h3>
            <button 
              onClick={onClose}
              className="text-gray-500 hover:text-gray-700"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="text-sm font-medium text-gray-500 mb-2">Основная информация</h4>
              <ul className="space-y-2">
                <li className="flex justify-between">
                  <span className="text-gray-600">ID:</span>
                  <span className="font-medium">{agent.id}</span>
                </li>
                <li className="flex justify-between">
                  <span className="text-gray-600">Тип модели:</span>
                  <span className="font-medium">{agent.model_type}</span>
                </li>
                <li className="flex justify-between">
                  <span className="text-gray-600">Статус:</span>
                  <span className={`font-medium ${
                    agent.status === 'active' ? 'text-green-600' : 
                    agent.status === 'training' ? 'text-yellow-600' : 
                    'text-red-600'
                  }`}>
                    {agent.status}
                  </span>
                </li>
                <li className="flex justify-between">
                  <span className="text-gray-600">Создан:</span>
                  <span className="font-medium">{new Date(agent.created_at).toLocaleString()}</span>
                </li>
              </ul>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="text-sm font-medium text-gray-500 mb-2">Производительность</h4>
              <ul className="space-y-2">
                <li className="flex justify-between">
                  <span className="text-gray-600">Точность:</span>
                  <span className="font-medium">
                    {agent.accuracy ? (agent.accuracy * 100).toFixed(2) + '%' : 'N/A'}
                  </span>
                </li>
                <li className="flex justify-between">
                  <span className="text-gray-600">Последнее обучение:</span>
                  <span className="font-medium">
                    {agent.last_trained ? new Date(agent.last_trained).toLocaleString() : 'N/A'}
                  </span>
                </li>
                <li className="flex justify-between">
                  <span className="text-gray-600">Время обучения:</span>
                  <span className="font-medium">
                    {agent.training_time ? agent.training_time + ' мин' : 'N/A'}
                  </span>
                </li>
                <li className="flex justify-between">
                  <span className="text-gray-600">Количество эпох:</span>
                  <span className="font-medium">{agent.epochs || 'N/A'}</span>
                </li>
              </ul>
            </div>
          </div>
          
          <div className="mb-6">
            <h4 className="text-sm font-medium text-gray-500 mb-2">Используемые фичи</h4>
            <div className="flex flex-wrap gap-2">
              {agent.features?.map((feature, index) => (
                <span 
                  key={index} 
                  className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded"
                >
                  {feature}
                </span>
              ))}
            </div>
          </div>
          
          <div className="mb-6">
            <h4 className="text-sm font-medium text-gray-500 mb-2">Описание</h4>
            <p className="text-gray-600">
              {agent.description || 'Описание отсутствует'}
            </p>
          </div>
          
          <div className="flex justify-end space-x-3">
            <button 
              onClick={onClose}
              className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
            >
              Закрыть
            </button>
            <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
              Переобучить
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentDetailsModal;