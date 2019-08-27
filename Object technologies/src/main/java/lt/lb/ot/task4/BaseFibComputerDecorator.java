/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package lt.lb.ot.task4;

/**
 *
 * @author Laimonas Beniušis
 */
public class BaseFibComputerDecorator extends AbstractFibComputerDecorator{
    
    protected FibComputer comp;
    
    public BaseFibComputerDecorator(FibComputer c){
        this.comp = c;
    }

    @Override
    protected FibComputer delegate() {
        return this.comp;
    }
    
}
