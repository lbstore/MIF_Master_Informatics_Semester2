/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package lt.lb.ot.task2.spring.config;

import java.util.Date;

/**
 *
 * @author Lemmin
 */
public interface OutputPrinter {

    void printDouble(Double d);

    void printInt(Integer i);

    void printString(String str);

    void printDate(Date date);

}
