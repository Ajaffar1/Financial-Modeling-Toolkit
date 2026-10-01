from finmodel import DCFModel
model = DCFModel(revenue=100_000_000, ebitda_margin=0.22, tax_rate=0.265,
                 wacc=0.09, terminal_growth=0.025).forecast(5)
print(f"Enterprise value: {model.enterprise_value():,.0f}")
print(f"Equity value: {model.equity_value():,.0f}")
print(model.audit())
print(model.scenario("Downside", {"revenue_growth": -0.05, "ebitda_margin": 0.20}).enterprise_value())
model.export_excel("Company_Model.xlsx")
