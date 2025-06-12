import api from './api';

export const get_coins = async (coinsData) => {
  const response = await api.get('/coins/get_coins/', coinsData);

  if (response.status !== 200) throw new Error('Ошибка загрузки монет');

  return response.data;
};

export const get_coin_time_line = async (coinsDataTimeLine) => {
  const params = {
    coin_id: coinsDataTimeLine.coin_id,
    timeframe: coinsDataTimeLine.timeframe,
    size_page: coinsDataTimeLine.size_page || 100  // Добавляем размер страницы
  };

  if (coinsDataTimeLine.last_timestamp) {
    params.last_timestamp = coinsDataTimeLine.last_timestamp;
  }
  
  const response = await api.get('/coins/get_coin/', {
    params: params, 
    headers: { 'Content-Type': 'application/json' }
  });

  if (response.status !== 200) throw new Error('Ошибка загрузки монет');

  return response.data;
};

// export const login = async (credentials) => {
//   const data = new URLSearchParams();
//   data.append('username', credentials.username);
//   data.append('password', credentials.password);
//   data.append('grant_type', 'password');
  
//   const response = await api.post('/auth/login_user/', data, {
//     headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
//   });
  
//   return response.data.access_token;
// };

// export const getCurrentUser = async () => {
//   const response = await api.get('auth/user/me/');
//   return response.data;
// };

// export const logout = () => {
//   localStorage.removeItem('access_token');
// };