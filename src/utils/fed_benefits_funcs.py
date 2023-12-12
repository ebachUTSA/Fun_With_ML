from bs4 import BeautifulSoup as bs

def getFedBennitsUrls(zipCode):
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

def parseFedBenefitsDataTable(soup):
    dataTable = soup.find('table',{'id':'DashboardNotificationsTable'})
    dataHeaderRow = dataTable.find('thead',{'id':'PlanTableHead'}).find('tr')
    dataRows = dataTable.find('tbody').find_all('tr')
    parseRow(dataHeaderRow)
    
    
def parseRow(soup):
    for th in soup.find_all('th'):
        # for div in th.find_all('div',{'class':'description'}):
        #     div.decompose()
        # headerValue = th.text.strip()
        descriptionDiv = th.find('div',{'class':'description'})
        if descriptionDiv is not None:
            headerValue = descriptionDiv.get('title')
        else:
            continue
        print(headerValue)
