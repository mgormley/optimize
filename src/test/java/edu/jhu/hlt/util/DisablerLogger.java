package edu.jhu.hlt.util;

import org.apache.log4j.Logger;


public class DisablerLogger {

    private Logger log;
    private boolean enabled = true;

    public DisablerLogger(Logger log) {
        this.log = log;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public void fatal(Object message) {
        if (enabled) { log.fatal(message); }
    }    
    
    public void error(Object message) {
        if (enabled) { log.error(message); }
    }
    
    public void warn(Object message) {
        if (enabled) { log.warn(message); }
    }
    
    public void info(Object message) {
        if (enabled) { log.info(message); }
    }
    
    public void debug(Object message) {
        if (enabled) { log.debug(message); }
    }

    public void trace(Object message) {
        if (enabled) { log.trace(message); }
    }
    
}
