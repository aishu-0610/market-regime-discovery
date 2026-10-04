import { useState, useEffect } from 'react'

function App() {
  const [status, setStatus] = useState('checking...')

  useEffect(() => {
    fetch('/api/health')
      .then((res) => res.json())
      .then((data) => setStatus(data.status))
      .catch(() => setStatus('unreachable'))
  }, [])

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-blue-500">Market Regime Discovery</h1>
      <p className="mt-2">API: {status}</p>
    </div>
  )
}

export default App