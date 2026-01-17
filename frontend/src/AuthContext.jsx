import React, { useState } from 'react';

export const AuthContext = React.createContext(null);

export const AuthProvider = ({ children }) => {
  const [auth, setAuth] = useState(() => {
    const token = localStorage.getItem('token');
    return token ? { token, user: null, loading: true } : { token: null, user: null, loading: false };
  });

  const login = (token) => {
    localStorage.setItem('token', token);
    setAuth(prev => ({ ...prev, token, loading: true }));
  };

  const logout = () => {
    localStorage.removeItem('token');
    setAuth({ token: null, user: null, loading: false });
  };

  return (
    <AuthContext.Provider value={{ auth, setAuth, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};
