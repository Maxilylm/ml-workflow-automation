function TabContainer({ activeTab, onTabChange, children }) {
  const tabs = ['Model Performance', 'Explore Data'];

  return (
    <div>
      <div className="flex border-b border-gray-700 mb-4">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => onTabChange(tab)}
            className={`px-4 py-2 font-medium transition-colors ${
              activeTab === tab
                ? 'text-white border-b-2 border-blue-500'
                : 'text-gray-400 hover:text-gray-300'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>
      {children}
    </div>
  );
}

export default TabContainer;
