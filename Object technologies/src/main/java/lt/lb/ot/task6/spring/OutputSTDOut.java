/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package lt.lb.ot.task6.spring;

import java.util.Date;
import lt.lb.commons.Log;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

/**
 *
 * @author Lemmin
 */
public class OutputSTDOut implements OutputPrinter{

    @Autowired
    @Qualifier("logSTD")
    protected Log log;
    
    
    
    @Qualifier("long")
    @Autowired
    private DateFormatter dateFormat;
    
    
    @Override
    public void printDouble(Double d) {
        Log.print(log, d);
    }

    @Override
    public void printInt(Integer i) {
        Log.print(log, i);
    }

    @Override
    public void printString(String str) {
        Log.print(log, str);
    }

    @Override
    public void printDate(Date date) {
        Log.print(log, dateFormat.format(date));
    }
    
    public Log getLog() {
        return log;
    }

    public void setLog(Log log) {
        this.log = log;
    }
    
    public void init(){
        Log.print(log,"INIT");
    }
    
    public void denit(){
        Log.print(log, "Close");
    }
    
}
