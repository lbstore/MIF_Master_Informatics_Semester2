/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package lt.lb.ot.task5.visitables;

import lt.lb.ot.task5.ExplicitVisitor;

/**
 *
 * @author Lemmin
 */
public class Cup implements Visitable {

    @Override
    public void accept(ExplicitVisitor visitor) {
        visitor.visit(this);
    }

}



