import React, { useState, useContext } from 'react';
import axios from 'axios';
import { AuthContext } from './AuthContext';
import { User, Lock, Mail, AlertCircle, ArrowRight, Loader2 } from 'lucide-react';

const Login = () => {
  const { login } = useContext(AuthContext);
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    email: ''
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      if (isLogin) {
        // Login Logic
        const params = new URLSearchParams();
        params.append('username', formData.username);
        params.append('password', formData.password);
        
        const response = await axios.post('/api/auth/token', params, {
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
        });
        
        login(response.data.access_token);
      } else {
        // Register Logic
        await axios.post('/api/auth/register', {
          username: formData.username,
          password: formData.password,
          email: formData.email
        });
        
        // Auto login after success or switch to login logic
        // For simplicity/security, let's login immediately
        const params = new URLSearchParams();
        params.append('username', formData.username);
        params.append('password', formData.password);
        
        const loginRes = await axios.post('/api/auth/token', params, {
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
        });
        login(loginRes.data.access_token);
      }
    } catch (err) {
      console.error(err);
      if (err.response) {
        setError(err.response.data.detail || '操作失败，请检查输入');
      } else {
        setError('无法连接到服务器，请联系管理员');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-slate-50 relative overflow-hidden">
      {/* Background Decorative Elements */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-100 rounded-full blur-3xl opacity-50" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-100 rounded-full blur-3xl opacity-50" />

      <div className="w-full max-w-md bg-white/80 backdrop-blur-xl rounded-2xl shadow-xl border border-white/20 overflow-hidden z-10 p-8 transform transition-all">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-slate-800 tracking-tight">
            {isLogin ? '欢迎回来' : '创建账户'}
          </h1>
          <p className="text-slate-500 mt-2">
            {isLogin ? '登录以继续使用 IMG2XML' : '注册即送 30 次免费额度'}
          </p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50/50 border border-red-100 text-red-600 rounded-xl flex items-start gap-3 text-sm animate-in fade-in slide-in-from-top-2">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <p>{error}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider ml-1">用户名</label>
            <div className="relative group">
              <User className="w-5 h-5 absolute left-3 top-3 text-slate-400 group-focus-within:text-indigo-500 transition-colors" />
              <input
                name="username"
                type="text"
                required
                value={formData.username}
                onChange={handleChange}
                className="w-full bg-slate-50 border border-slate-200 text-slate-800 text-sm rounded-xl py-3 pl-10 pr-4 outline-none focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/10 transition-all font-medium placeholder:text-slate-400"
                placeholder="请输入用户名"
              />
            </div>
          </div>

          {!isLogin && (
            <div className="space-y-1.5 animate-in fade-in slide-in-from-top-2">
              <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider ml-1">邮箱</label>
              <div className="relative group">
                <Mail className="w-5 h-5 absolute left-3 top-3 text-slate-400 group-focus-within:text-indigo-500 transition-colors" />
                <input
                  name="email"
                  type="email"
                  required={!isLogin}
                  value={formData.email}
                  onChange={handleChange}
                  className="w-full bg-slate-50 border border-slate-200 text-slate-800 text-sm rounded-xl py-3 pl-10 pr-4 outline-none focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/10 transition-all font-medium placeholder:text-slate-400"
                  placeholder="name@example.com"
                />
              </div>
            </div>
          )}

          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider ml-1">密码</label>
            <div className="relative group">
              <Lock className="w-5 h-5 absolute left-3 top-3 text-slate-400 group-focus-within:text-indigo-500 transition-colors" />
              <input
                name="password"
                type="password"
                required
                value={formData.password}
                onChange={handleChange}
                className="w-full bg-slate-50 border border-slate-200 text-slate-800 text-sm rounded-xl py-3 pl-10 pr-4 outline-none focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/10 transition-all font-medium placeholder:text-slate-400"
                placeholder="••••••••"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3.5 rounded-xl shadow-lg shadow-indigo-600/20 active:scale-[0.98] transition-all flex items-center justify-center gap-2 mt-6"
          >
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <>
                {isLogin ? '登录' : '立即注册'}
                <ArrowRight className="w-5 h-5" />
              </>
            )}
          </button>
        </form>

        <div className="mt-6 text-center">
          <button
            type="button"
            onClick={() => {
              setIsLogin(!isLogin);
              setError('');
              setFormData({ username: '', password: '', email: '' });
            }}
            className="text-sm font-medium text-slate-500 hover:text-indigo-600 transition-colors"
          >
            {isLogin ? (
              <>还没有账号？ <span className="underline decoration-2 underline-offset-4 decoration-indigo-200 hover:decoration-indigo-500">去注册</span></>
            ) : (
              <>已有账号？ <span className="underline decoration-2 underline-offset-4 decoration-indigo-200 hover:decoration-indigo-500">去登录</span></>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Login;
