import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import useAuth from './hooks/useAuth';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ProfilePage from './pages/ProfilePage';
import spinner from './img/pageload-spinner.gif';
import logo from './img/logo.webp';

const FullPageSpinner = () => (
  <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-900 to-purple-900">
    <div className="mb-8 animate-pulse">
      <img 
        src={logo} 
        alt="AI Trading Logo" 
        className="w-40 h-auto"
      />
    </div>
    <div className="flex flex-col items-center">
      <img 
        src={spinner} 
        alt="Loading..." 
        className="w-16 h-16"
      />
      <p className="mt-4 text-xl text-blue-200 font-semibold">
        Проверка авторизации...
      </p>
    </div>
  </div>
);

const App = () => {
  const { user, isAuthenticated, loading, login, logout } = useAuth();

  if (loading) {
    return <FullPageSpinner />;
  }

  return (
    <Router>
      <Routes>
        <Route path="/profile_page" element={
          isAuthenticated 
            ? <ProfilePage user={user} onLogout={logout} /> 
            : <Navigate to="/login" />
        } />
        
        <Route path="/login" element={
          !isAuthenticated 
            ? <LoginPage onLogin={login} /> 
            : <Navigate to="/profile_page" />
        } />
        
        <Route path="/register" element={
          !isAuthenticated 
            ? <RegisterPage /> 
            : <Navigate to="/profile_page" />
        } />
        
        <Route path="/" element={
          <Navigate to={isAuthenticated ? "/profile_page" : "/login"} />
        } />
      </Routes>
    </Router>
  );
};

export default App;