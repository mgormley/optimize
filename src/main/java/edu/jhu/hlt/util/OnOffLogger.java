package edu.jhu.hlt.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class OnOffLogger {

    private Logger log;
    private boolean enabled = true;

    public OnOffLogger(Logger log) {
        this.log = log;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
    
    public void error(String message) {
        if (enabled) { log.error(message); }
    }
    
    public void warn(String message) {
        if (enabled) { log.warn(message); }
    }
    
    public void info(String message) {
        if (enabled) { log.info(message); }
    }
    
    public void debug(String message) {
        if (enabled) { log.debug(message); }
    }

    public void trace(String message) {
        if (enabled) { log.trace(message); }
    }
    
    public boolean isTraceEnabled() {
        return log.isTraceEnabled();
    }

    public void trace(String message, Throwable t) {
        log.trace(message, t);
    }
    
}
