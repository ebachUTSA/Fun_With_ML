def getFedBenniesUrl(zipCode):
    payPeriods =   {
                        'c':'biweekly',
                        'd':'every four weeks',
                        'b':'semi-monthly',
                        'a':'monthly'
                    }
    empTypes = {
                    'a':'Federal-And-US-Postal' 
                    ,'g':'Federal-Deposit-Insurance-Corporation'
                    ,'e':'Certain-Temporary-Employees'
                    ,'k':'Tribal-Employee'
                    ,'b':'Annuitant'
                    ,'f':'Former-Spouse-Enrollee'
                    ,'t':'Temporary-Continuation-of-Coverage-TCC'
                    ,'d':'Workers-Compensation-Recipient'
                }
    urlParams = {
                    'a':'abcd'
                    ,'g':'abcd'
                    ,'e':'abcd'
                    ,'k':'a'
                    ,'b':'a'
                    ,'f':'a'
                    ,'t':'a'
                    ,'d':'d'
    }
    results = []
    for empType,payPeriod_avails in urlParams.items():
        for payPeriod in payPeriod_avails:
            results.append((f"https://www.opm.gov/healthcare-insurance/healthcare/plan-information/compare-plans/fehb/Plans?ZipCode={zipCode}&empType={empType}&payPeriod={payPeriod}",zipCode,empTypes[empType],payPeriods[payPeriod]))
    return results #returns a list of tuples